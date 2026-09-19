"""Differentiable, display-referred color utilities (D65 sRGB)."""

from __future__ import annotations

import torch
from torch import Tensor


def srgb_to_linear(rgb: Tensor) -> Tensor:
    # Clamp the unused power branch too: torch.where evaluates both branches.
    nonlinear = ((rgb + 0.055).clamp_min(0) / 1.055).pow(2.4)
    return torch.where(rgb <= 0.04045, rgb / 12.92, nonlinear)


def linear_to_srgb(rgb: Tensor) -> Tensor:
    nonlinear = 1.055 * rgb.clamp_min(1e-12).pow(1 / 2.4) - 0.055
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, nonlinear)


def luminance(rgb: Tensor) -> Tensor:
    """Rec.709 luminance; pass linear RGB for physical luminance."""
    return (rgb * rgb.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)


def linear_rgb_to_lab(rgb: Tensor) -> Tensor:
    """D65 CIELAB, useful for differentiable evaluation (not a gamut mapping)."""
    matrix = rgb.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.einsum("ij,bjhw->bihw", matrix, rgb)
    xyz = xyz / rgb.new_tensor([0.95047, 1.0, 1.08883])[None, :, None, None]
    delta = 6 / 29
    f = torch.where(xyz > delta**3, xyz.clamp_min(1e-12).pow(1 / 3), xyz / (3 * delta**2) + 4 / 29)
    return torch.cat(
        (116 * f[:, 1:2] - 16, 500 * (f[:, 0:1] - f[:, 1:2]), 200 * (f[:, 1:2] - f[:, 2:3])), dim=1
    )


def srgb_to_lab(rgb: Tensor) -> Tensor:
    return linear_rgb_to_lab(srgb_to_linear(rgb))
