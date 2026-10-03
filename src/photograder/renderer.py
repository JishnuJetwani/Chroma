"""Resolution-independent differentiable rendering without custom kernels.

LUTs use explicit corner gathers because MPS does not implement 5D grid_sample.
Tiled rendering predicts nothing per tile: tiles share one recipe and overlap by
the local tone operator's support, then discard their halo.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .color import linear_to_srgb, luminance, srgb_to_linear
from .recipes import EditRecipe


def identity_lut(size: int = 33, *, device=None, dtype=torch.float32) -> Tensor:
    axis = torch.linspace(0, 1, size, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=0)


def apply_lut(image: Tensor, luts: Tensor, extrapolate: bool = False) -> Tensor:
    """Trilinear interpolation of B3DDD tables; gradients reach tables and input."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected BCHW RGB image")
    if luts.ndim == 4:
        luts = luts.unsqueeze(0)
    batch, _, height, width = image.shape
    size = luts.shape[-1]
    if luts.shape[1:] != (3, size, size, size):
        raise ValueError("LUT must have shape B x 3 x D x D x D")
    if luts.shape[0] == 1 and batch != 1:
        luts = luts.expand(batch, -1, -1, -1, -1)
    source = image if extrapolate else image.clamp(0, 1)
    coords = source.reshape(batch, 3, -1) * (size - 1)
    lower = coords.floor().long().clamp(min=0, max=size - 2)
    fraction = coords - lower.to(coords.dtype)
    flat = luts.reshape(batch, 3, -1)
    result = torch.zeros_like(coords)
    for r in range(2):
        for g in range(2):
            for b in range(2):
                offsets = (r, g, b)
                index = (lower[:, 0] + r) * size * size + (lower[:, 1] + g) * size + lower[:, 2] + b
                weight = torch.ones_like(fraction[:, 0])
                for channel, offset in enumerate(offsets):
                    weight = weight * (fraction[:, channel] if offset else 1 - fraction[:, channel])
                values = torch.gather(flat, 2, index[:, None].expand(-1, 3, -1))
                result = result + values * weight[:, None]
    return result.reshape(batch, 3, height, width)


def box_filter(image: Tensor, radius: int) -> Tensor:
    """Constant-work separable box filter with replicate boundary handling."""
    if radius <= 0:
        return image
    kernel = 2 * radius + 1
    horizontal = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    sums = F.pad(horizontal.cumsum(-1), (1, 0, 0, 0))
    horizontal = (sums[..., kernel:] - sums[..., :-kernel]) / kernel
    vertical = F.pad(horizontal, (0, 0, radius, radius), mode="replicate")
    sums = F.pad(vertical.cumsum(-2), (0, 0, 1, 0))
    return (sums[..., kernel:, :] - sums[..., :-kernel, :]) / kernel


def guided_coefficients(guide: Tensor, field: Tensor, radius: int = 2, eps: float = 0.001):
    guide = F.interpolate(guide, size=field.shape[-2:], mode="bilinear", align_corners=False)
    mean_i, mean_p = (box_filter(guide, radius), box_filter(field, radius))
    variance = (box_filter(guide * guide, radius) - mean_i.square()).clamp_min(0)
    covariance = box_filter(guide * field, radius) - mean_i * mean_p
    slope = covariance / (variance + eps)
    intercept = mean_p - slope * mean_i
    return (box_filter(slope, radius), box_filter(intercept, radius))


def _sample_region(field: Tensor, bounds: tuple[int, int, int, int], size: tuple[int, int]) -> Tensor:
    """align_corners=False bilinear resize restricted to a global output rectangle."""
    top, bottom, left, right = bounds
    out_h, out_w = size
    in_h, in_w = field.shape[-2:]
    ys = (
        (torch.arange(top, bottom, device=field.device, dtype=field.dtype) + 0.5) * in_h / out_h - 0.5
    ).clamp(0, in_h - 1)
    xs = (
        (torch.arange(left, right, device=field.device, dtype=field.dtype) + 0.5) * in_w / out_w - 0.5
    ).clamp(0, in_w - 1)
    y0, x0 = (ys.floor().long(), xs.floor().long())
    y1, x1 = ((y0 + 1).clamp(max=in_h - 1), (x0 + 1).clamp(max=in_w - 1))
    fy, fx = ((ys - y0)[:, None], (xs - x0)[None, :])
    flat = field.flatten(2)
    output = None
    for yy, wy in ((y0, 1 - fy), (y1, fy)):
        for xx, wx in ((x0, 1 - fx), (x1, fx)):
            index = (yy[:, None] * in_w + xx[None, :]).flatten()
            sampled = torch.index_select(flat, 2, index).reshape(
                field.shape[0], field.shape[1], bottom - top, right - left
            )
            value = sampled * (wy * wx)[None, None]
            output = value if output is None else output + value
    return output


def _prepare_field(field: Tensor | None, guide: Tensor | None, upsampling: str):
    if field is None:
        return None
    if upsampling == "guided" and guide is not None:
        return guided_coefficients(guide, field)
    return (None, field)


def _field_at(prepared, guidance: Tensor, bounds, size):
    if prepared is None:
        return torch.zeros_like(guidance)
    slope, intercept = prepared
    value = _sample_region(intercept, bounds, size)
    if slope is not None:
        value = value + _sample_region(slope, bounds, size) * guidance
    return value.clamp(-1, 1)


def exposure_field(local_input: Tensor, recipe: EditRecipe, upsampling: str | None = None) -> Tensor:
    """Exposure stops actually applied at full resolution, before strength.

    Pass the image entering the local operator (already globally graded when
    applicable). This uses the same guide-dependent interpolation and bounds as
    rendering, so supervised map losses do not target a different upsampler.
    """
    method = upsampling or recipe.metadata.get("upsampling", "guided")
    if method not in ("guided", "bilinear"):
        raise ValueError("upsampling must be guided or bilinear")
    height, width = local_input.shape[-2:]
    prepared = _prepare_field(recipe.exposure, recipe.guide, method)
    return _field_at(prepared, luminance(local_input), (0, height, 0, width), (height, width))


def _local(image, prepared_exposure, prepared_contrast, bounds, size, strength, radius):
    linear = srgb_to_linear(image)
    light = luminance(linear).clamp_min(1e-06)
    log_light = torch.log2(light)
    slope, intercept = guided_coefficients(log_light, log_light, radius=radius, eps=0.05)
    detail = log_light - (slope * log_light + intercept)
    guide = luminance(image)
    exposure = _field_at(prepared_exposure, guide, bounds, size) * strength
    log_contrast = _field_at(prepared_contrast, guide, bounds, size) * strength
    ratio = torch.exp2(exposure + (torch.exp2(log_contrast) - 1) * detail)
    return linear_to_srgb(linear * ratio)


def render_with_aux(
    image: Tensor,
    recipe: EditRecipe,
    grade_strength: float = 1.0,
    local_strength: float = 1.0,
    tile_size: int | None = None,
    upsampling: str | None = None,
    _clip_output: bool = True,
    _compute_gamut: bool = True,
):
    """Render a recipe and expose unclipped sRGB for gamut regularization.

    grade_strength controls the global transform,
    while local_strength scales exposure stops and log-contrast. Zero for both
    is an exact bypass. Clip only at the final output.
    """
    if not 0 <= grade_strength <= 1 or not 0 <= local_strength <= 1:
        raise ValueError("Strengths must be between 0 and 1")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected BCHW RGB image")
    if grade_strength == 0 and local_strength == 0:
        return (image, {"pre_gamut": image, "gamut_penalty": image.new_zeros(())})
    height, width = image.shape[-2:]
    full_size = (height, width)
    method = upsampling or recipe.metadata.get("upsampling", "guided")
    if method not in ("guided", "bilinear"):
        raise ValueError("upsampling must be guided or bilinear")
    prepared_exp = _prepare_field(recipe.exposure, recipe.guide, method)
    prepared_con = _prepare_field(recipe.log_contrast, recipe.guide, method)
    has_local = local_strength != 0 and (prepared_exp is not None or prepared_con is not None)
    radius = max(1, int(math.ceil(max(height, width) / 256 * 5))) if has_local else 0
    size = tile_size or max(height, width)
    if size <= 0:
        raise ValueError("tile_size must be positive")
    order = recipe.metadata.get("order", "global_first")

    def grade(patch):
        if recipe.luts is None or grade_strength == 0:
            return patch
        graded = apply_lut(patch, recipe.luts, extrapolate=True)
        return patch + grade_strength * (graded - patch)

    rows = []
    for top in range(0, height, size):
        columns = []
        bottom = min(top + size, height)
        for left in range(0, width, size):
            right = min(left + size, width)
            halo = 2 * radius
            bounds = (
                max(0, top - halo),
                min(height, bottom + halo),
                max(0, left - halo),
                min(width, right + halo),
            )
            y0, y1, x0, x1 = bounds
            patch = image[..., y0:y1, x0:x1]
            if order == "local_first":
                if has_local:
                    patch = _local(
                        patch, prepared_exp, prepared_con, bounds, full_size, local_strength, radius
                    )
                patch = grade(patch)
            else:
                patch = grade(patch)
                if has_local:
                    patch = _local(
                        patch, prepared_exp, prepared_con, bounds, full_size, local_strength, radius
                    )
            core = patch[..., top - y0 : bottom - y0, left - x0 : right - x0]
            columns.append(core)
        rows.append(torch.cat(columns, dim=-1))
    output = torch.cat(rows, dim=-2)
    raw = output
    penalty = (
        (F.relu(-raw).square() + F.relu(raw - 1).square()).mean() if _compute_gamut else image.new_zeros(())
    )
    if _clip_output:
        output = raw.clamp(0, 1)
    return (output, {"pre_gamut": raw, "gamut_penalty": penalty})


def render(
    image: Tensor,
    recipe: EditRecipe,
    grade_strength: float = 1.0,
    local_strength: float = 1.0,
    tile_size: int | None = None,
    upsampling: str | None = None,
) -> Tensor:
    return render_with_aux(
        image, recipe, grade_strength, local_strength, tile_size, upsampling, _compute_gamut=False
    )[0]
