"""Experimental photographic treatment for bright, veiled JPEGs.

This is a bounded tone-control policy, not a flare detector or a reconstruction
of clipped pixels. Analyze once; the same recipe and scalars serve every tile.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .color import linear_to_srgb, srgb_to_linear
from .recipes import EditRecipe


def _smoothstep(value: Tensor, low: float, high: float) -> Tensor:
    t = ((value - low) / (high - low)).clamp(0, 1)
    return t.square() * (3 - 2 * t)


def prepare_backlight_recipe(recipe: EditRecipe, preview: Tensor) -> EditRecipe:
    """Derive an opt-in treatment without altering the trained recipe/weights.

    The low percentile is a conservative black-floor hint, not a measured veil.
    No fixed sky/person coordinates or a photograph-specific mask are used.
    """
    if preview.ndim != 4 or preview.shape[0] != 1 or preview.shape[1] != 3:
        raise ValueError("Backlight analysis expects one BCHW RGB preview")
    if recipe.stages or recipe.rgb_residual is not None or recipe.bilateral_grid is not None:
        raise ValueError("Backlight treatment currently supports LUT and local-map models")
    from .lut_safety import smooth_lut_residual

    result = recipe.detach().to(preview.device)
    # New assignments only: tensors shared by detach() must not be changed in place.
    if result.luts is not None:
        result.luts = smooth_lut_residual(result.luts, kernel_size=5, method="gaussian", sigma=1.0)
    if result.exposure is not None:
        result.exposure = result.exposure * 0.5
    if result.log_contrast is not None:
        result.log_contrast = result.log_contrast * 0.5
    linear = srgb_to_linear(preview)
    # Quantile is computed once on the small CPU preview for MPS portability.
    floor = float(torch.quantile(linear.amin(1).flatten().float().cpu(), 0.05))
    veil = min(0.06, max(0.0, floor * 0.4))
    result.metadata = dict(result.metadata)
    result.metadata["backlight"] = {
        "version": 1,
        "linear_black_floor_hint": floor,
        "veil": veil,
        "highlight_start": 0.78,
        "highlight_end": 0.98,
        "local_multiplier": 0.5,
        "toe_softness": 0.025,
        "experimental": True,
    }
    return result


def finish_backlight(
    source: Tensor, rendered: Tensor, policy: dict, grade_strength: float, local_strength: float
) -> Tensor:
    """Pointwise, white-preserving tone control with a soft shadow toe.

    Must receive the original source pixels, even after the learned global LUT.
    Both zero strengths are an exact bypass, including quantized PNG exports.
    """
    if grade_strength == 0 and local_strength == 0:
        return source
    if policy.get("version") != 1:
        raise ValueError("Unsupported backlight recipe version")
    protect = _smoothstep(source.amin(1, keepdim=True), policy["highlight_start"], policy["highlight_end"])
    veil = float(policy["veil"]) * local_strength
    if veil > 0:
        linear = srgb_to_linear(rendered.clamp(0, 1))
        tau = float(policy["toe_softness"])
        # Smooth black subtraction, pinned at 0 and 1, never hard-clips shadows.
        zero = F.softplus(linear.new_tensor(-veil / tau)) * tau
        one = F.softplus(linear.new_tensor((1 - veil) / tau)) * tau
        mapped = (F.softplus((linear - veil) / tau) * tau - zero) / (one - zero)
        rendered = linear_to_srgb(mapped.clamp_min(0))
    # Convex blending keeps source-white interiors flat and avoids gray islands.
    return (rendered * (1 - protect) + source * protect).clamp(0, 1)
