"""Small style-conditioned CNNs that predict executable photo edit recipes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .color import luminance
from .recipes import EditRecipe
from .renderer import apply_lut, identity_lut, render


@dataclass
class ModelConfig:
    architecture: str = "local"
    styles: tuple[str, ...] = ("Expert C",)
    preview_size: int = 256
    lut_size: int = 33
    num_luts: int = 8
    map_size: int = 64
    upsampling: str = "guided"
    order: str = "global_first"

    def __post_init__(self):
        self.styles = tuple(self.styles)
        if self.architecture not in ("fixed", "global", "local"):
            raise ValueError("Unknown architecture: " + self.architecture)
        if not self.styles or len(set(self.styles)) != len(self.styles):
            raise ValueError("styles must be a nonempty sequence of unique names")
        if self.preview_size < 32 or self.map_size < 4 or self.lut_size < 2 or (self.num_luts < 1):
            raise ValueError("Invalid preview, map, or LUT size")
        if self.order not in ("global_first", "local_first"):
            raise ValueError("order must be global_first or local_first")
        if self.upsampling not in ("guided", "bilinear"):
            raise ValueError("upsampling must be guided or bilinear")


def _preview(image: Tensor, size: int):
    height, width = image.shape[-2:]
    scale = size / max(height, width)
    new_h, new_w = (max(1, round(height * scale)), max(1, round(width * scale)))
    resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top, left = ((size - new_h) // 2, (size - new_w) // 2)
    return (
        F.pad(resized, (left, size - new_w - left, top, size - new_h - top), mode="replicate"),
        (top, left, new_h, new_w),
    )


def _unpad(field: Tensor, crop, preview_size: int, map_size: int) -> Tensor:
    top, left, height, width = crop
    field = F.interpolate(field, size=(preview_size, preview_size), mode="bilinear", align_corners=False)
    field = field[..., top : top + height, left : left + width]
    output_size = (
        max(1, round(map_size * height / max(height, width))),
        max(1, round(map_size * width / max(height, width))),
    )
    return F.interpolate(field, size=output_size, mode="bilinear", align_corners=False)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvBlock(3, 32, 2), ConvBlock(32, 64, 2), ConvBlock(64, 128, 2), ConvBlock(128, 128, 2)]
        )

    def forward(self, image):
        features = []
        for block in self.blocks:
            image = block(image)
            features.append(image)
        return features


class SpatialDecoder(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.up1 = ConvBlock(256, 64)
        self.up2 = ConvBlock(128, 32)
        self.fuse = ConvBlock(32 + 32 + 16 + 1, 32)
        self.output = nn.Conv2d(32, outputs, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, style, guide):
        f0, f1, f2, f3 = features
        x = self.up1(
            torch.cat((F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2), 1)
        )
        x = self.up2(
            torch.cat((F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1), 1)
        )
        early = F.adaptive_avg_pool2d(f0, x.shape[-2:])
        guide = F.interpolate(guide, size=x.shape[-2:], mode="bilinear", align_corners=False)
        style = style[:, :, None, None].expand(-1, -1, *x.shape[-2:])
        return self.output(self.fuse(torch.cat((x, early, style, guide), dim=1)))


class LUTBank(nn.Module):
    def __init__(self, count, size):
        super().__init__()
        identity = identity_lut(size)
        values = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3, 1, 1, 1)
        coefficients = torch.sin(values * 2.39996 + 0.3)
        coefficients -= coefficients.mean(dim=0, keepdim=True)
        variation = 0.02 * coefficients * identity[None] * (1 - identity[None])
        self.tables = nn.Parameter(identity[None].repeat(count, 1, 1, 1, 1) + variation)

    def forward(self, weights):
        return torch.einsum("bn,ncijk->bcijk", weights, self.tables)

    def regularization(self):
        smooth, monotonic = (self.tables.new_zeros(()), self.tables.new_zeros(()))
        for axis in (2, 3, 4):
            diff = self.tables.diff(dim=axis)
            monotonic = monotonic + F.relu(-diff).square().mean()
            if diff.shape[axis] > 1:
                smooth = smooth + diff.diff(dim=axis).square().mean()
        return {"lut_smoothness": smooth, "lut_monotonicity": monotonic}


class RecipeModel(nn.Module):
    def forward(self, image: Tensor, style: Tensor | None = None, valid_mask: Tensor | None = None) -> Tensor:
        return render(image, self.analyze(image, style, valid_mask=valid_mask))

    def _style(self, image, style):
        if style is None:
            style = torch.zeros(image.shape[0], dtype=torch.long, device=image.device)
        elif isinstance(style, int):
            style = torch.full((image.shape[0],), style, dtype=torch.long, device=image.device)
        else:
            style = style.to(device=image.device, dtype=torch.long)
        if style.shape != (image.shape[0],):
            raise ValueError("style must contain one index per image")
        valid = ((style >= 0) & (style < len(self.config.styles))).all()
        if image.device.type == "cuda":
            torch._assert_async(valid, "style index outside configured styles")
        elif not bool(valid):
            raise ValueError("style index outside configured styles")
        return style

    def _metadata(self):
        return {
            "version": 1,
            "architecture": self.config.architecture,
            "upsampling": self.config.upsampling,
            "order": self.config.order,
        }


class FixedLUTModel(RecipeModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bank = LUTBank(len(config.styles), config.lut_size)
        with torch.no_grad():
            self.bank.tables.copy_(identity_lut(config.lut_size)[None])

    def analyze(self, image, style=None, bypass_global=False, valid_mask=None):
        style = self._style(image, style)
        return EditRecipe(
            luts=None if bypass_global else self.bank.tables[style],
            source_size=tuple(image.shape[-2:]),
            metadata=self._metadata(),
        )

    def regularization(self):
        return self.bank.regularization()


class GradingModel(RecipeModel):
    """Shared feature extractor; global and spatial branches have distinct jobs."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder()
        self.style_embedding = nn.Embedding(len(config.styles), 16)
        nn.init.normal_(self.style_embedding.weight, std=0.02)
        self.has_global = config.architecture in ("global", "local")
        self.has_local = config.architecture == "local"
        if self.has_global:
            self.bank = LUTBank(config.num_luts, config.lut_size)
            self.mixer = nn.Linear(128 + 16, config.num_luts)
            nn.init.zeros_(self.mixer.weight)
            nn.init.zeros_(self.mixer.bias)
        if self.has_local:
            self.decoder = SpatialDecoder(2)

    def analyze(self, image, style=None, bypass_global=False, valid_mask=None):
        with torch.autocast(device_type=image.device.type, enabled=False):
            return self._analyze_fp32(image.float(), style, bypass_global, valid_mask)

    def _analyze_fp32(self, image, style=None, bypass_global=False, valid_mask=None):
        style = self._style(image, style)
        preview, crop = _preview(image, self.config.preview_size)
        features = self.encoder(preview)
        embedding = self.style_embedding(style)
        recipe = EditRecipe(source_size=tuple(image.shape[-2:]), metadata=self._metadata())
        guide_image = preview
        if self.has_global and (not bypass_global):
            pooled = features[-1].float().mean((-2, -1))
            conditioning = [pooled, embedding]
            weights = self.mixer(torch.cat(conditioning, 1)).softmax(-1)
            recipe.luts = self.bank(weights)
            if self.config.order == "global_first":
                guide_image = apply_lut(preview, recipe.luts)
        guide = luminance(guide_image)
        recipe.guide = _unpad(guide, crop, self.config.preview_size, self.config.map_size)
        if self.has_local:
            output = self.decoder(features, embedding, guide)
            output = _unpad(output.float(), crop, self.config.preview_size, self.config.map_size)
            recipe.exposure = torch.tanh(output[:, :1])
            recipe.log_contrast = torch.tanh(output[:, 1:2])
        return recipe

    def regularization(self):
        if self.has_global:
            return self.bank.regularization()
        zero = next(self.parameters()).new_zeros(())
        return {"lut_smoothness": zero, "lut_monotonicity": zero}


def build_model(config=None):
    config = config or ModelConfig()
    if isinstance(config, dict):
        config = ModelConfig(**config)
    return FixedLUTModel(config) if config.architecture == "fixed" else GradingModel(config)


def initialize_from_model(target: RecipeModel, source: RecipeModel) -> dict:
    """Transfer compatible weights, matching learned style rows by name.

    Global LUT bank indices denote shared basis functions, not editor IDs.
    Only fixed LUTs and explicit style embeddings have a style-indexed axis.
    Newly introduced styles retain their fresh initialization. This helper is
    for initialization of a new experiment, never optimizer-state resumption.
    """

    def style_parameters(model):
        result = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and name.split(".")[-1] == "style_embedding":
                result.add(name + ".weight")
            if isinstance(module, FixedLUTModel):
                result.add((name + "." if name else "") + "bank.tables")
        return result

    target_state, source_state = (target.state_dict(), source.state_dict())
    target_style_keys, source_style_keys = (style_parameters(target), style_parameters(source))
    source_indices = {name: index for index, name in enumerate(source.config.styles)}
    matching = [
        (index, source_indices[name], name)
        for index, name in enumerate(target.config.styles)
        if name in source_indices
    ]
    compatible, copied_rows, skipped = ({}, {}, [])
    for key, value in source_state.items():
        if key not in target_state:
            skipped.append(key)
            continue
        destination = target_state[key]
        is_style = key in source_style_keys or key in target_style_keys
        if is_style:
            if (
                key not in source_style_keys
                or key not in target_style_keys
                or destination.shape[1:] != value.shape[1:]
                or (not matching)
            ):
                skipped.append(key)
                continue
            remapped = destination.detach().clone()
            for new_index, old_index, _ in matching:
                remapped[new_index].copy_(
                    value[old_index].to(device=destination.device, dtype=destination.dtype)
                )
            compatible[key] = remapped
            copied_rows[key] = [
                {"style": name, "source_index": old, "target_index": new} for new, old, name in matching
            ]
        elif value.shape == destination.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    if not compatible:
        raise ValueError("No compatible weights in initialization checkpoint.")
    target.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": len(compatible),
        "style_rows_copied": copied_rows,
        "new_styles": [name for name in target.config.styles if name not in source_indices],
        "source_styles": list(source.config.styles),
        "target_styles": list(target.config.styles),
        "skipped_tensors": skipped,
        "fresh_target_tensors": [key for key in target_state if key not in compatible],
    }
