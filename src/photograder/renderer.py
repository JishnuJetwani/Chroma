"""Resolution-independent differentiable rendering without custom kernels.

LUTs use explicit corner gathers because MPS does not implement 5D grid_sample.
Every tile applies the same image-level color table.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


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


def render_with_aux(
    image, recipe, grade_strength=1.0, tile_size=None, _clip_output=True, _compute_gamut=True
):
    """Apply one global LUT to a photo, with strength and gamut diagnostics."""
    if not 0 <= grade_strength <= 1:
        raise ValueError("Strengths must be between 0 and 1")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected BCHW RGB image")
    if grade_strength == 0 or recipe.luts is None:
        return (image, dict(pre_gamut=image, gamut_penalty=image.new_zeros(())))
    size = tile_size or max(image.shape[-2:])
    if size <= 0:
        raise ValueError("tile_size must be positive")
    rows = []
    for y in range(0, image.shape[-2], size):
        columns = []
        for x in range(0, image.shape[-1], size):
            patch = image[..., y : y + size, x : x + size]
            columns.append(patch + grade_strength * (apply_lut(patch, recipe.luts, extrapolate=True) - patch))
        rows.append(torch.cat(columns, -1))
    raw = torch.cat(rows, -2)
    penalty = (F.relu(-raw) + F.relu(raw - 1)).mean() if _compute_gamut else raw.new_zeros(())
    return (raw.clamp(0, 1) if _clip_output else raw, dict(pre_gamut=raw, gamut_penalty=penalty))


def render(image, recipe, grade_strength=1.0, tile_size=None):
    return render_with_aux(image, recipe, grade_strength, tile_size, _compute_gamut=False)[0]
