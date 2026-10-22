"""Opt-in research losses and additive-light supervision; not wired into training.

These helpers do not change a model, checkpoint, global RNG, or training policy.
All loss tensors retain gradients. A corruption target is supervised only where
``available`` is nonzero; clipped values do not provide texture-recovery labels.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from .color import linear_to_srgb, luminance, srgb_to_linear


def _batched_luts(luts: Tensor) -> Tensor:
    if not isinstance(luts, Tensor) or not luts.is_floating_point():
        raise ValueError("luts must be a floating-point tensor")
    if luts.ndim == 4:
        luts = luts.unsqueeze(0)
    if (
        luts.ndim != 5
        or luts.shape[0] < 1
        or luts.shape[1] != 3
        or luts.shape[-1] < 2
        or len(set(luts.shape[-3:])) != 1
    ):
        raise ValueError("Expected LUTs shaped 3DDD or B3DDD, with D >= 2")
    # Keep arithmetic at least FP32, including under an external CNN autocast.
    return luts.float() if luts.dtype in (torch.float16, torch.bfloat16) else luts


def lut_shape_losses(luts: Tensor, *, normalize_derivatives: bool = True) -> dict[str, Tensor]:
    """Unsquared monotonicity hinge and squared curvature over all RGB axes.

    With normalization, first differences are divided by h=1/(D-1) and second
    differences by h**2. Affine tables have zero curvature. Each result is the
    mean over the three axes and all batch/channel/cell entries; e.g., a reversed
    identity LUT has monotonicity 1/3. This constrains cross-channel derivatives
    too, which can restrict valid color transforms. It is a soft loss, not a
    monotonicity guarantee. D=2 has no curvature samples and returns graph-zero.

    ``normalize_derivatives=False`` returns unscaled finite differences for an
    explicit comparison with legacy losses (the hinge still is not squared).
    """
    luts = _batched_luts(luts)
    h = 1 / (luts.shape[-1] - 1) if normalize_derivatives else 1.0
    first = [luts.diff(dim=axis) for axis in (2, 3, 4)]
    monotonicity = torch.stack([F.relu(-d / h).mean() for d in first]).mean()
    if luts.shape[-1] > 2:
        curvature = torch.stack(
            [(d.diff(dim=axis) / h**2).square().mean() for axis, d in zip((2, 3, 4), first, strict=False)]
        ).mean()
    else:
        curvature = luts.sum() * 0
    return {"lut_monotonicity": monotonicity, "lut_curvature": curvature}


def lut_ramp_losses(
    luts: Tensor,
    *,
    samples: int = 257,
    near_neutral_offset: float = 0.025,
    normalize_derivatives: bool = True,
    interpolator: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> dict[str, Tensor]:
    """Probe a complete neutral ramp plus six affine near-neutral RGB rays.

    ``interpolator(image, luts)`` must return unclamped B3HW colors and retain
    gradients to its tables; the default is this repository's trilinear LUT
    renderer. The input has H=7 rays and W=``samples`` positions. Near-neutral
    rays shift one RGB channel's black or white endpoint by the given offset,
    so every input ray stays affine and within [0,1] without endpoint clipping.
    Offset zero evaluates just the full neutral ray.

    Monotonicity is mean ReLU of the negative Rec.709-weighted *encoded RGB*
    slope (a tone-order diagnostic, not physical luminance). Curvature is the
    mean squared second difference of RGB output minus an identically rendered
    identity LUT. That reference cancels interpolation roundoff at identity;
    its second derivative is zero in exact arithmetic. Gradients flow through
    the predicted output. With normalization the divisors are dt and dt**2,
    where dt=1/(samples-1). Piecewise-linear interpolation has slope jumps, so
    normalized ramp curvature depends on sample density: compare a fixed grid.
    Dense rays are diagnostics/soft constraints, not a whole-gamut guarantee.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
        raise ValueError("samples must be an integer >= 3")
    if not 0 <= near_neutral_offset < 0.5:
        raise ValueError("near_neutral_offset must be in [0, 0.5)")
    from .renderer import apply_lut, identity_lut

    luts = _batched_luts(luts)
    interpolate = apply_lut if interpolator is None else interpolator
    offsets = luts.new_zeros((1, 3))
    if near_neutral_offset:
        axes = torch.eye(3, dtype=luts.dtype, device=luts.device) * near_neutral_offset
        offsets = torch.cat((offsets, axes, -axes), dim=0)
    t = torch.linspace(0, 1, samples, dtype=luts.dtype, device=luts.device)
    start, end = offsets.clamp_min(0), 1 + offsets.clamp_max(0)
    rays = start[..., None] + (end - start)[..., None] * t
    image = rays.permute(1, 0, 2)[None].expand(luts.shape[0], -1, -1, -1)
    output = interpolate(image, luts)
    if output.shape != image.shape:
        raise ValueError("interpolator must preserve the B3HW image shape")
    reference_luts = identity_lut(luts.shape[-1], dtype=luts.dtype, device=luts.device)
    reference = interpolate(image, reference_luts[None].expand_as(luts))
    if reference.shape != image.shape:
        raise ValueError("interpolator must preserve the B3HW image shape")
    dt = 1 / (samples - 1) if normalize_derivatives else 1.0
    monotonicity = F.relu(-luminance(output).diff(dim=-1) / dt).mean()
    curvature = ((output - reference).diff(dim=-1).diff(dim=-1) / dt**2).square().mean()
    return {"ramp_monotonicity": monotonicity, "ramp_curvature": curvature}


def additive_veil_corruption(
    images: Tensor,
    veil: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compose a supplied nonnegative RGB veil in linear light, with masks.

    ``images`` is finite floating B3HW sRGB in [0,1]. ``veil`` is linear-light
    intensity broadcastable to B3HW (for a color vector use shape 1x3x1x1).
    Field generation and random seeds belong to the caller. ``valid_mask`` is
    optional B1HW in [0,1]; positive values permit corruption and also weight
    the returned ``available`` supervision mask. Padding/privacy zeros remain
    untouched. Outside exact support, output equals input bit for bit.

    Return values:
      image: corrupted sRGB, clamped to [0,1].
      target: original clean tensor, with gradients preserved.
      support: bool B1HW, any positive valid veil channel.
      clipped: bool B3HW, a positive veil pushes that channel to/above linear 1.
      available: float B3HW, valid weight where no new clipping occurred.

    ``target`` is intentionally retained unchanged. The caller MUST use
    ``available`` for reconstruction: ``clipped`` channels have no exact
    texture-recovery supervision. Existing saturation without added veil is
    not labeled newly clipped. No global headroom scaling erases the fault.
    The binary availability decision is nondifferentiable; image formation
    remains differentiable on recoverable pixels and consumes no RNG state.
    """
    if (
        not isinstance(images, Tensor)
        or not images.is_floating_point()
        or images.ndim != 4
        or images.shape[1] != 3
        or min(images.shape) < 1
    ):
        raise ValueError("images must be floating B3HW sRGB")
    if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
        raise ValueError("images must be finite sRGB values in [0,1]")
    if not isinstance(veil, Tensor) or not veil.is_floating_point():
        raise ValueError("veil must be a nonnegative floating-point linear-light tensor")
    if not bool(torch.isfinite(veil).all()) or bool((veil < 0).any()):
        raise ValueError("veil must be finite and nonnegative")
    if valid_mask is None:
        valid = torch.ones_like(images[:, :1])
    else:
        if valid_mask.shape != (images.shape[0], 1, *images.shape[-2:]):
            raise ValueError("valid_mask must be B1HW")
        valid = valid_mask.to(device=images.device, dtype=images.dtype)
        if not bool(torch.isfinite(valid).all()) or bool(((valid < 0) | (valid > 1)).any()):
            raise ValueError("valid_mask must be finite and in [0,1]")
    try:
        field = torch.broadcast_to(veil.to(device=images.device, dtype=images.dtype), images.shape)
    except RuntimeError as error:
        raise ValueError("veil must broadcast to the B3HW image shape") from error
    field = torch.where(valid > 0, field, torch.zeros_like(field))
    changed = field > 0
    linear = srgb_to_linear(images) + field
    clipped = changed & (linear >= 1)
    encoded = linear_to_srgb(linear.clamp(0, 1)).clamp(0, 1)
    output = torch.where(changed, encoded, images)
    return {
        "image": output,
        "target": images,
        "support": changed.any(1, keepdim=True),
        "clipped": clipped,
        "available": valid.expand_as(images) * (~clipped).to(images.dtype),
    }
