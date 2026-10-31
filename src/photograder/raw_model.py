"""Scene-linear RAW grading with editable, monotone-tone recipes.

Input is demosaiced, white-balanced linear sRGB, not a sensor mosaic or an
encoded JPEG. Negative CCM values and values above one are retained until the
final luminance-preserving display gamut map. Decoder metadata lives outside
this module. All recipe and renderer arithmetic is FP32; no custom kernels.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .color import linear_to_srgb


@dataclass
class RawModelConfig:
    preview_size: int = 256
    map_size: int = 64
    tone_knots: int = 33
    styles: tuple[str, ...] = ("Expert C",)
    enable_veil: bool = False
    max_exposure_ev: float = 4.0
    max_wb_log2: float = 1.0
    max_local_exposure_ev: float = 1.5
    max_local_log_contrast: float = 0.5
    max_veil: float = 0.15

    def __post_init__(self):
        self.styles = tuple(self.styles)
        if self.preview_size < 32 or self.map_size < 4 or self.tone_knots < 3:
            raise ValueError("RAW preview/map/curve sizes must be >= 32/4/3")
        if not self.styles or len(self.styles) != len(set(self.styles)):
            raise ValueError("styles must be nonempty and unique")
        for name in (
            "max_exposure_ev",
            "max_wb_log2",
            "max_local_exposure_ev",
            "max_local_log_contrast",
            "max_veil",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class RawEditRecipe:
    exposure_ev: Tensor  # B1; relative to decoder development
    white_balance_log2: Tensor  # B3; zero-mean log2 gains
    color_matrix: Tensor  # B33; preserves luminance and gray axis
    saturation: Tensor  # B1; positive
    tone_y: Tensor  # BK; increasing, first=0 and last=1
    local_exposure_ev: Tensor  # B1hw
    local_log_contrast: Tensor  # B1hw; log2 multiplier of local detail
    guide_log2: Tensor  # B1hw; source log luminance detail base
    veil: Tensor | None = None  # B3hw; nonnegative scene-linear light
    source_size: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def _map(self, fn):
        return RawEditRecipe(
            **{
                item.name: fn(value)
                if isinstance(value := getattr(self, item.name), Tensor)
                else dict(value)
                if isinstance(value, dict)
                else value
                for item in fields(self)
            }
        )

    def to(self, *args, **kwargs):
        return self._map(lambda value: value.to(*args, **kwargs))

    def detach(self):
        return self._map(lambda value: value.detach())

    def state_dict(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_state_dict(cls, state):
        state = dict(state)
        if state.get("source_size") is not None:
            state["source_size"] = tuple(state["source_size"])
        return cls(**state)


def _image(image: Tensor) -> Tensor:
    if not isinstance(image, Tensor) or not image.is_floating_point():
        raise ValueError("RAW image must be a floating-point B3HW tensor")
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape) < 1:
        raise ValueError("RAW image must have nonempty B3HW shape")
    return image.float()


def encode_raw_preview(image: Tensor) -> Tensor:
    """Signed log encoding with fixed scale, preserving absolute exposure cues."""
    return image.sign() * torch.log1p(16 * image.abs()) / math.log(17)


def _raw_luminance(rgb: Tensor) -> Tensor:
    # Equivalent Rec.709 weights, with an exactly neutral result on equal RGB.
    # This avoids gray-axis roundoff becoming visible in an almost-flat curve.
    red, green, blue = rgb[:, :1], rgb[:, 1:2], rgb[:, 2:3]
    return green + 0.2126 * (red - green) + 0.0722 * (blue - green)


def _shoulder(y: Tensor) -> Tensor:
    positive = y.clamp_min(0)
    # The reciprocal form retains monotone floating-point behavior in bright
    # regions; the quotient retains precision near zero.
    return torch.where(positive > 1, 1 - 1 / (1 + positive), positive / (1 + positive))


def monotone_tone_knots(logits: Tensor) -> Tensor:
    """Positive increments imply a monotone piecewise-linear luminance curve.

    The fixed positive floor also prevents numerical collapse under extreme
    logits. Zero logits give the identity in the shoulder coordinate.
    """
    increments = F.softplus(logits.float().clamp(-20, 20) + math.log(math.expm1(1))) + 1e-3
    cumulative = increments.cumsum(-1)
    return F.pad(cumulative / cumulative[:, -1:], (1, 0))


def apply_tone_curve(coordinate: Tensor, knots: Tensor) -> Tensor:
    """Interpolate a BK curve at B1HW coordinates in [0,1]."""
    position = coordinate.clamp(0, 1) * (knots.shape[-1] - 1)
    lower = position.floor().long().clamp(0, knots.shape[-1] - 2)
    fraction = position - lower
    a = knots.gather(1, lower.flatten(1)).reshape_as(coordinate)
    b = knots.gather(1, (lower + 1).flatten(1)).reshape_as(coordinate)
    return a + fraction * (b - a)


def _display_gamut_map(rgb: Tensor, tone_luminance: Tensor, source_luminance: Tensor) -> Tensor:
    """Compress chroma toward gray only as far as the display gamut requires."""
    chroma = (rgb - source_luminance) * (tone_luminance / source_luminance.clamp_min(1e-6))
    positive = chroma.amax(1, keepdim=True).clamp_min(1e-6)
    negative = (-chroma).amax(1, keepdim=True).clamp_min(1e-6)
    scale = torch.minimum(
        torch.ones_like(tone_luminance),
        torch.minimum((1 - tone_luminance) / positive, tone_luminance / negative),
    )
    return (tone_luminance + scale * chroma).clamp(0, 1)


def baseline_raw(image: Tensor) -> Tensor:
    """Fixed neutral development: Y/(1+Y), gamut compression, then sRGB.

    Thus scene-linear gray 1 develops to sRGB ~0.735, and 2 remains brighter.
    This is the explicitly defined zero-strength preview, not a camera JPEG.
    """
    with torch.autocast(device_type=image.device.type, enabled=False):
        rgb = _image(image)
        source_y = _raw_luminance(rgb)
        display_y = _shoulder(source_y)
        return linear_to_srgb(_display_gamut_map(rgb, display_y, source_y))


def _sample_map(field: Tensor, full_size, region) -> Tensor:
    """Global-coordinate bilinear interpolation identical for full/tiled render."""
    height, width = full_size
    top, left, rows, cols = region
    ys = (
        (torch.arange(top, top + rows, device=field.device, dtype=field.dtype) + 0.5)
        * field.shape[-2]
        / height
        - 0.5
    ).clamp(0, field.shape[-2] - 1)
    xs = (
        (torch.arange(left, left + cols, device=field.device, dtype=field.dtype) + 0.5)
        * field.shape[-1]
        / width
        - 0.5
    ).clamp(0, field.shape[-1] - 1)
    y0, x0 = ys.floor().long(), xs.floor().long()
    y1, x1 = (y0 + 1).clamp_max(field.shape[-2] - 1), (x0 + 1).clamp_max(field.shape[-1] - 1)
    fy, fx = (ys - y0)[None, None, :, None], (xs - x0)[None, None, None, :]
    upper = field.index_select(-2, y0)
    lower = field.index_select(-2, y1)
    a = upper.index_select(-1, x0).lerp(upper.index_select(-1, x1), fx)
    b = lower.index_select(-1, x0).lerp(lower.index_select(-1, x1), fx)
    return a.lerp(b, fy)


def _render_region(
    image: Tensor, recipe: RawEditRecipe, strength: float, full_size, region, exposure_bias_ev=0.0
):
    def sample(value):
        return _sample_map(value, full_size, region)

    corrected = image
    if recipe.veil is not None:
        corrected = corrected - strength * sample(recipe.veil)
    y = _raw_luminance(corrected)
    detail = (y.clamp_min(1e-6).log2() - sample(recipe.guide_log2)).clamp(-4, 4)
    local_ev = strength * sample(recipe.local_exposure_ev)
    local_ev = local_ev + (torch.exp2(strength * sample(recipe.local_log_contrast)) - 1) * detail
    corrected = corrected * torch.exp2(
        local_ev + strength * recipe.exposure_ev[:, :, None, None] + exposure_bias_ev
    )
    corrected = corrected * torch.exp2(strength * recipe.white_balance_log2[:, :, None, None])
    identity = torch.eye(3, device=image.device, dtype=image.dtype)[None]
    matrix = identity + strength * (recipe.color_matrix - identity)
    corrected = torch.einsum("bij,bjhw->bihw", matrix, corrected)
    y = _raw_luminance(corrected)
    saturation = 1 + strength * (recipe.saturation[:, :, None, None] - 1)
    corrected = y + saturation * (corrected - y)
    coordinate = _shoulder(y)
    knots = torch.linspace(0, 1, recipe.tone_y.shape[-1], device=image.device)[None]
    tone = knots + strength * (recipe.tone_y - knots)
    display_y = apply_tone_curve(coordinate, tone)
    linear = _display_gamut_map(corrected, display_y, y)
    pre_gamut = corrected * (display_y / y.clamp_min(1e-6))
    return linear_to_srgb(linear), {
        "linear_display": linear,
        "tone_luminance": display_y,
        "pre_gamut": pre_gamut,
        "gamut_penalty": F.relu(-pre_gamut) + F.relu(pre_gamut - 1),
    }


def render_raw(
    image: Tensor,
    recipe: RawEditRecipe,
    *,
    strength: float = 1.0,
    tile_size: int | None = None,
    return_aux: bool = False,
    exposure_bias_ev: float = 0.0,
):
    """Reuse a preview recipe at native resolution, optionally with output tiles.

    Strength blends parameters toward neutral development. It must be in [0,1]
    so the curve stays monotonic. No image clamp at one occurs before tonemapping.
    Tiles sample maps in whole-image coordinates and need no image halo.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("RAW strength must be finite and in [0,1]")
    if not math.isfinite(exposure_bias_ev) or not -8 <= exposure_bias_ev <= 8:
        raise ValueError("exposure_bias_ev must be finite and in [-8,8]")
    if tile_size is not None and (
        isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1
    ):
        raise ValueError("tile_size must be a positive integer")
    with torch.autocast(device_type=image.device.type, enabled=False):
        image = _image(image)
        recipe = recipe.to(device=image.device, dtype=torch.float32)
        if recipe.exposure_ev.shape[0] != image.shape[0]:
            raise ValueError("Recipe batch size must match RAW image")
        if strength == 0:
            output = baseline_raw(image * (2**exposure_bias_ev))
            return (output, {}) if return_aux else output
        height, width = image.shape[-2:]
        tile = tile_size or max(height, width)
        output_rows, auxiliary_rows = [], []
        for top in range(0, height, tile):
            output_cols, auxiliary_cols = [], []
            for left in range(0, width, tile):
                patch = image[..., top : top + tile, left : left + tile]
                output, auxiliary = _render_region(
                    patch, recipe, strength, (height, width), (top, left, *patch.shape[-2:]), exposure_bias_ev
                )
                output_cols.append(output)
                if return_aux:
                    auxiliary_cols.append(auxiliary)
            output_rows.append(torch.cat(output_cols, -1))
            if return_aux:
                auxiliary_rows.append(
                    {key: torch.cat([item[key] for item in auxiliary_cols], -1) for key in auxiliary_cols[0]}
                )
        output = torch.cat(output_rows, -2)
        if return_aux:
            return output, {
                key: torch.cat([item[key] for item in auxiliary_rows], -2) for key in auxiliary_rows[0]
            }
        return output


def render_raw_tile(
    image: Tensor,
    recipe: RawEditRecipe,
    *,
    full_size: tuple[int, int],
    origin: tuple[int, int],
    strength: float = 1.0,
    exposure_bias_ev: float = 0.0,
) -> Tensor:
    """Render one native-image tile without moving or allocating the full image.

    ``image`` contains only the tile. ``full_size`` and ``origin`` refer to the
    complete oriented source image; maps must never restart at a tile boundary.
    The caller can transfer each tile to an accelerator and immediately store
    its quantized result on CPU, bounding accelerator memory at export time.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("RAW strength must be finite and in [0,1]")
    if not math.isfinite(exposure_bias_ev) or not -8 <= exposure_bias_ev <= 8:
        raise ValueError("exposure_bias_ev must be finite and in [-8,8]")
    image = _image(image)
    height, width = full_size
    top, left = origin
    if height < 1 or width < 1 or top < 0 or left < 0:
        raise ValueError("Tile origin and full image size must be valid")
    if top + image.shape[-2] > height or left + image.shape[-1] > width:
        raise ValueError("Tile extends beyond the full image")
    if strength == 0:
        return baseline_raw(image * (2**exposure_bias_ev))
    with torch.autocast(device_type=image.device.type, enabled=False):
        recipe = recipe.to(device=image.device, dtype=torch.float32)
        return _render_region(
            image, recipe, strength, full_size, (top, left, *image.shape[-2:]), exposure_bias_ev
        )[0]


class _Block(nn.Sequential):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1),
            nn.GroupNorm(8, outgoing),
            nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(8, outgoing),
            nn.SiLU(),
        )


class RawGradingModel(nn.Module):
    """One small CNN predicts global controls and restrained spatial controls."""

    def __init__(self, config: RawModelConfig | dict | None = None):
        super().__init__()
        self.config = RawModelConfig(**config) if isinstance(config, dict) else config or RawModelConfig()
        self.encoder = nn.ModuleList(
            [_Block(3, 24, 2), _Block(24, 48, 2), _Block(48, 96, 2), _Block(96, 128, 2)]
        )
        self.style_embedding = nn.Embedding(len(self.config.styles), 16)
        self.global_head = nn.Sequential(
            nn.Linear(128 + 16 + 12, 128), nn.SiLU(), nn.Linear(128, 14 + self.config.tone_knots - 1)
        )
        self.up1, self.up2 = _Block(128 + 96, 64), _Block(64 + 48, 32)
        self.local_head = nn.Sequential(
            _Block(32 + 16 + 1, 32), nn.Conv2d(32, 5 if self.config.enable_veil else 2, 1)
        )
        nn.init.normal_(self.style_embedding.weight, std=0.02)
        for head in (self.global_head[-1], self.local_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self._cnn_autocast = "none"

    def set_cnn_autocast(self, mode="none"):
        if mode not in ("none", "bf16"):
            raise ValueError("CNN autocast must be none or bf16")
        self._cnn_autocast = mode
        return self

    def _scope(self, image):
        if self._cnn_autocast == "none":
            return nullcontext()
        if image.device.type not in ("cpu", "cuda"):
            raise ValueError("CNN BF16 requires CPU or CUDA; use none on MPS")
        return torch.autocast(image.device.type, dtype=torch.bfloat16)

    def _style(self, image, style):
        if style is None or isinstance(style, int):
            index = torch.full((image.shape[0],), style or 0, device=image.device, dtype=torch.long)
        else:
            index = torch.as_tensor(style, device=image.device, dtype=torch.long)
        if index.shape != (image.shape[0],):
            raise ValueError("style must contain one index per image")
        valid = ((index >= 0) & (index < len(self.config.styles))).all()
        if image.device.type == "cuda":
            torch._assert_async(valid, "style index outside configured styles")
        elif not bool(valid):
            raise ValueError("style index outside configured styles")
        return index

    def analyze(self, image: Tensor, style=None, valid_mask: Tensor | None = None) -> RawEditRecipe:
        with torch.autocast(device_type=image.device.type, enabled=False):
            return self._analyze(_image(image), style, valid_mask)

    def _analyze(self, image, style, valid_mask):
        source_size = tuple(image.shape[-2:])
        scale = self.config.preview_size / max(source_size)
        size = tuple(max(1, round(value * scale)) for value in source_size)
        # Replicate padding only serves the CNN. Stats and maps use real content.
        preview = (
            image
            if tuple(image.shape[-2:]) == size
            else F.interpolate(image, size=size, mode="bilinear", align_corners=False, antialias=True)
        )
        encoded = encode_raw_preview(preview)
        pad_h, pad_w = self.config.preview_size - size[0], self.config.preview_size - size[1]
        top, left = pad_h // 2, pad_w // 2
        padded = F.pad(encoded, (left, pad_w - left, top, pad_h - top), mode="replicate")
        weight = (
            torch.ones_like(preview[:, :1])
            if valid_mask is None
            else F.interpolate(
                valid_mask.to(device=image.device, dtype=image.dtype), size=size, mode="nearest"
            )
        )
        if weight.shape != (image.shape[0], 1, *size):
            raise ValueError("valid_mask must be B1HW")
        mass = weight.sum((-2, -1)).clamp_min(1)
        mean = (encoded * weight).sum((-2, -1)) / mass
        std = (
            (((encoded - mean[:, :, None, None]).square() * weight).sum((-2, -1)) / mass)
            .clamp_min(1e-8)
            .sqrt()
        )
        log_y = _raw_luminance(preview).clamp_min(1e-6).log2()
        y_mean = (log_y * weight).sum((-2, -1)) / mass
        y_std = (
            (((log_y - y_mean[:, :, None, None]).square() * weight).sum((-2, -1)) / mass)
            .clamp_min(1e-8)
            .sqrt()
        )
        bright = ((preview > 1).float() * weight).sum((-2, -1)) / mass
        negative = ((preview < 0).any(1, keepdim=True).float() * weight).sum((-2, -1)) / mass
        stats = torch.cat((mean, std, y_mean / 8, y_std / 8, bright, negative), dim=1)
        embedding = self.style_embedding(self._style(image, style))
        with self._scope(image):
            features = []
            x = padded
            for block in self.encoder:
                x = block(x)
                features.append(x)
        context = torch.cat((features[-1].float().mean((-2, -1)), embedding, stats), 1)
        global_values = self.global_head(context)
        guide = F.pad(log_y / 8, (left, pad_w - left, top, pad_h - top), mode="replicate")
        with self._scope(image):
            x = self.up1(
                torch.cat(
                    (
                        F.interpolate(
                            features[3], size=features[2].shape[-2:], mode="bilinear", align_corners=False
                        ),
                        features[2],
                    ),
                    1,
                )
            )
            x = self.up2(
                torch.cat(
                    (
                        F.interpolate(x, size=features[1].shape[-2:], mode="bilinear", align_corners=False),
                        features[1],
                    ),
                    1,
                )
            )
            local = self.local_head(
                torch.cat(
                    (
                        x,
                        embedding[:, :, None, None].expand(-1, -1, *x.shape[-2:]),
                        F.interpolate(guide, size=x.shape[-2:], mode="bilinear", align_corners=False),
                    ),
                    1,
                )
            )
        local = F.interpolate(
            local.float(), size=(self.config.preview_size,) * 2, mode="bilinear", align_corners=False
        )[..., top : top + size[0], left : left + size[1]]
        map_size = tuple(max(1, round(self.config.map_size * value / max(size))) for value in size)
        local = F.interpolate(local, size=map_size, mode="bilinear", align_corners=False)
        # Fixed pooling supports arbitrary aspect ratios on MPS; adaptive area
        # pooling requires divisible dimensions there in the supported Torch.
        base = F.avg_pool2d(log_y, kernel_size=9, stride=1, padding=4, count_include_pad=False)
        guide_log2 = F.interpolate(base, size=map_size, mode="bilinear", align_corners=False)
        exposure = self.config.max_exposure_ev * global_values[:, :1].tanh()
        # Centering before tanh keeps every gain bounded; centering after could exceed the bound.
        wb_raw = global_values[:, 1:4]
        wb = 0.5 * self.config.max_wb_log2 * (wb_raw.tanh() - wb_raw.tanh().mean(1, keepdim=True))
        identity = torch.eye(3, device=image.device)[None]
        luma = image.new_tensor([0.2126, 0.7152, 0.0722])
        projection = identity - torch.ones(3, 1, device=image.device) * luma[None]
        residual = global_values[:, 4:13].reshape(-1, 3, 3).tanh()
        matrix = identity + 0.15 * (projection @ residual @ projection)
        saturation = 1 + 0.5 * global_values[:, 13:14].tanh()
        tone_y = monotone_tone_knots(global_values[:, 14:])
        veil = self.config.max_veil * local[:, 2:5].tanh().clamp_min(0) if self.config.enable_veil else None
        return RawEditRecipe(
            exposure,
            wb,
            matrix,
            saturation,
            tone_y,
            self.config.max_local_exposure_ev * local[:, :1].tanh(),
            self.config.max_local_log_contrast * local[:, 1:2].tanh(),
            guide_log2,
            veil,
            source_size,
            {
                "version": 1,
                "input_domain": "scene-linear-srgb",
                "output_domain": "display-srgb",
                "tone_coordinate": "Y/(1+Y)",
                "zero_strength": "neutral-shoulder-development",
            },
        )

    def forward(self, image: Tensor, style=None, valid_mask=None):
        return render_raw(image, self.analyze(image, style, valid_mask))


def _masked_mean(value, mask):
    weights = torch.broadcast_to(mask, value.shape)
    return (value * weights).sum() / weights.sum().clamp_min(1)


def raw_supervised_loss(
    prediction: Tensor,
    target: Tensor,
    recipe: RawEditRecipe,
    valid_mask: Tensor | None = None,
    available_mask: Tensor | None = None,
    *,
    gradient_weight: float = 0.05,
    map_smoothness_weight: float = 0.002,
    adjustment_weight: float = 0.0005,
    tone_curvature_weight: float = 0.0001,
    highlight_weight: float = 0.05,
    noop_prediction: Tensor | None = None,
    noop_target: Tensor | None = None,
    noop_weight: float = 0.1,
) -> dict[str, Tensor]:
    """Display-sRGB supervision; unavailable clipped channels contribute no recovery loss.

    ``available_mask`` must already be conservative in output RGB coordinates:
    a clipped sensor channel can affect every output channel through the CCM.
    Optional no-op pairs must be genuinely scene-linear inputs developed to a
    known target; never pass a display-sRGB target straight back as RAW input.
    Returned ``total`` is differentiable and all named unweighted terms are logged.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("prediction and target must be matching B3HW tensors")
    mask = torch.ones_like(target[:, :1])
    for supplied in (valid_mask, available_mask):
        if supplied is not None:
            if supplied.shape not in (target.shape, target[:, :1].shape):
                raise ValueError("loss masks must be B1HW or B3HW")
            mask = mask * supplied.to(device=target.device, dtype=target.dtype)
    error = prediction - target
    terms = {"rgb_l1": _masked_mean(error.abs(), mask)}
    gradients = []
    for axis in (-1, -2):
        if target.shape[axis] > 1:
            a = [slice(None)] * 4
            b = [slice(None)] * 4
            a[axis], b[axis] = slice(1, None), slice(None, -1)
            edge_mask = torch.minimum(mask[tuple(a)], mask[tuple(b)])
            gradients.append(_masked_mean(error.diff(dim=axis).abs(), edge_mask))
    terms["gradient"] = torch.stack(gradients).mean() if gradients else error.sum() * 0
    spatial = [recipe.local_exposure_ev, recipe.local_log_contrast]
    if recipe.veil is not None:
        spatial.append(recipe.veil)
    smooth = [
        value.diff(dim=axis).abs().mean() for value in spatial for axis in (-1, -2) if value.shape[axis] > 1
    ]
    terms["map_smoothness"] = torch.stack(smooth).mean() if smooth else error.sum() * 0
    terms["adjustment"] = torch.stack([value.square().mean() for value in spatial]).mean()
    slope = recipe.tone_y.diff(dim=-1) * (recipe.tone_y.shape[-1] - 1)
    terms["tone_curvature"] = slope.diff(dim=-1).square().mean()
    bright_neutral = (target.amin(1, keepdim=True) > 0.85).to(target.dtype)
    terms["highlight"] = _masked_mean(error.abs(), mask * bright_neutral)
    terms["noop"] = error.sum() * 0
    if (noop_prediction is None) != (noop_target is None):
        raise ValueError("noop_prediction and noop_target must be supplied together")
    if noop_prediction is not None:
        if noop_prediction.shape != target.shape or noop_target.shape != target.shape:
            raise ValueError("no-op tensors must match target shape")
        terms["noop"] = _masked_mean((noop_prediction - noop_target).abs(), mask)
    terms["total"] = (
        terms["rgb_l1"]
        + gradient_weight * terms["gradient"]
        + map_smoothness_weight * terms["map_smoothness"]
        + adjustment_weight * terms["adjustment"]
        + tone_curvature_weight * terms["tone_curvature"]
        + highlight_weight * terms["highlight"]
        + noop_weight * terms["noop"]
    )
    return terms


def additive_linear_veil(
    image: Tensor, veil: Tensor, *, clip_level: float | Tensor | None = None, valid_mask: Tensor | None = None
) -> dict[str, Tensor]:
    """Deterministically add supplied light in the actual scene-linear domain.

    There is no assumed saturation at RGB=1 after white balance/CCM. With no
    declared clip_level, values above one remain available. An explicit clipping
    level is a synthetic RGB clipping experiment, not a sensor clipping model.
    Newly clipped channels are marked; availability conservatively excludes all
    RGB channels at affected pixels because color mixing propagates uncertainty.
    """
    image = _image(image)
    if not isinstance(veil, Tensor) or not veil.is_floating_point():
        raise ValueError("veil must be a floating-point nonnegative tensor")
    if not bool(torch.isfinite(veil).all()) or bool((veil < 0).any()):
        raise ValueError("veil must be finite and nonnegative")
    try:
        field = torch.broadcast_to(veil.to(image), image.shape)
    except RuntimeError as error:
        raise ValueError("veil must broadcast to B3HW") from error
    valid = torch.ones_like(image[:, :1]) if valid_mask is None else valid_mask.to(image)
    if valid.shape != image[:, :1].shape:
        raise ValueError("valid_mask must be B1HW")
    field = torch.where(valid > 0, field, torch.zeros_like(field))
    output = image + field
    clipped = torch.zeros_like(image, dtype=torch.bool)
    if clip_level is not None:
        level = torch.as_tensor(clip_level, device=image.device, dtype=image.dtype)
        if not bool(torch.isfinite(level).all()) or bool((level <= 0).any()):
            raise ValueError("clip_level must be finite and positive")
        clipped = (field > 0) & (output >= level)
        output = torch.where(field > 0, torch.minimum(output, level), image)
    return {
        "image": output,
        "target": image,
        "veil": field,
        "support": (field > 0).any(1, keepdim=True),
        "clipped": clipped,
        "available": valid * (~clipped.any(1, keepdim=True)).to(image.dtype),
    }
