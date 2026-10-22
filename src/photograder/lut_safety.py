"""Isolated LUT experiments; these do not change the production renderer.

Tetrahedra share each cube's R=G=B diagonal, as in the ACES CLF specification.
Neither interpolation nor residual smoothing guarantees arbitrary LUTs are
monotonic. Use the diagnostics and image comparisons before adopting a result.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .renderer import apply_lut, identity_lut


def _batched_luts(luts: Tensor) -> Tensor:
    if luts.ndim not in (4, 5):
        raise ValueError("LUT must have shape 3 x D x D x D or B x 3 x D x D x D")
    tables = luts.unsqueeze(0) if luts.ndim == 4 else luts
    size = tables.shape[-1]
    if size < 2 or tables.shape[1:] != (3, size, size, size) or tables.shape[0] < 1:
        raise ValueError("LUT must contain nonempty RGB cubes with D >= 2")
    if not tables.is_floating_point():
        raise ValueError("LUT values must be floating point")
    return tables


def apply_tetrahedral_lut(image: Tensor, luts: Tensor, extrapolate: bool = False) -> Tensor:
    """Interpolate RGB tables with four explicit gathers, preserving gradients.

    Inputs follow ``renderer.apply_lut``: BCHW RGB and 3DDD/B3DDD LUTs with
    R/G/B indexing and optional shared-table broadcasting. No output clamp.
    Normally input RGB is clamped to [0,1]. With ``extrapolate=True``, select
    the boundary cube and sort its *unclamped* fractional coordinates. The
    corresponding tetrahedron's affine function continues outside the cube;
    negative barycentric weights are intentional. This preserves affine LUTs
    for intermediate out-of-range colors, including identity. Gradients are
    piecewise defined at grid/tetrahedron boundaries, as in trilinear LUTs.
    """
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[0], *image.shape[-2:]) < 1:
        raise ValueError("Expected nonempty BCHW RGB image")
    tables = _batched_luts(luts)
    if not image.is_floating_point():
        raise ValueError("Image values must be floating point")
    if tables.device != image.device or tables.dtype != image.dtype:
        raise ValueError("Image and LUT must have the same device and floating dtype")
    batch, _, height, width = image.shape
    if tables.shape[0] == 1:
        tables = tables.expand(batch, -1, -1, -1, -1)
    elif tables.shape[0] != batch:
        raise ValueError("LUT batch must be one or match the image batch")
    size = tables.shape[-1]
    source = image if extrapolate else image.clamp(0, 1)
    coordinates = source.reshape(batch, 3, -1) * (size - 1)
    lower = coordinates.floor().long().clamp(0, size - 2)
    fraction = coordinates - lower.to(coordinates.dtype)
    ordered, axes = fraction.sort(dim=1, descending=True)
    base = lower[:, 0] * size * size + lower[:, 1] * size + lower[:, 2]
    strides = axes.new_tensor([size * size, size, 1])[axes]
    vertices = (
        base,
        base + strides[:, 0],
        base + strides[:, 0] + strides[:, 1],
        base + size * size + size + 1,
    )
    weights = (1 - ordered[:, 0], ordered[:, 0] - ordered[:, 1], ordered[:, 1] - ordered[:, 2], ordered[:, 2])
    flat = tables.reshape(batch, 3, -1)
    result = torch.zeros_like(coordinates)
    for vertex, weight in zip(vertices, weights, strict=False):
        values = torch.gather(flat, 2, vertex[:, None].expand(-1, 3, -1))
        result = result + values * weight[:, None]
    return result.reshape(batch, 3, height, width)


def smooth_lut_residual(
    luts: Tensor, kernel_size: int = 3, method: str = "gaussian", sigma: float = 1.0, passes: int = 1
) -> Tensor:
    """Return identity plus a separably smoothed residual, without mutation.

    Replicate-pad the residual rather than the table itself: an identity LUT
    has a zero residual and remains exactly identity, including its endpoints.
    Arbitrary LUT endpoints are not pinned. No clipping or monotonic projection
    is performed. Separate 1D convolutions avoid a 3D-convolution dependency on
    MPS. The returned tensor retains the input shape, device, dtype, and graph.
    """
    tables = _batched_luts(luts)
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    if method not in ("gaussian", "box"):
        raise ValueError("method must be gaussian or box")
    if not isinstance(passes, int) or passes < 0:
        raise ValueError("passes must be a nonnegative integer")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if passes == 0 or kernel_size == 1:
        return luts.clone()
    size = tables.shape[-1]
    identity = identity_lut(size, device=tables.device, dtype=tables.dtype)[None]
    residual = tables - identity
    radius = kernel_size // 2
    if method == "box":
        kernel = tables.new_ones(kernel_size)
    else:
        offsets = torch.arange(-radius, radius + 1, device=tables.device, dtype=tables.dtype)
        kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
    for _ in range(passes):
        for axis in (2, 3, 4):
            lines = residual.movedim(axis, -1)
            shape = lines.shape
            lines = F.pad(lines.reshape(-1, 1, size), (radius, radius), mode="replicate")
            residual = F.conv1d(lines, kernel).reshape(shape).movedim(-1, axis)
    result = identity + residual
    return result[0] if luts.ndim == 4 else result


@torch.inference_mode()
def diagnose_neutral_ramp(
    luts: Tensor,
    samples: int = 4097,
    start: float = 0.0,
    end: float = 1.0,
    interpolation: str = "trilinear",
    slope_tolerance: float = 1e-5,
) -> dict:
    """JSON-friendly, unclipped neutral-ramp slopes and endpoint diagnostics.

    A nonnegative result on this ramp says nothing about off-neutral colors.
    Derivatives are finite differences divided by input spacing, not output
    code differences. Reporting intentionally synchronizes/copies to CPU.
    """
    tables = _batched_luts(luts)
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    if not 0 <= start < end <= 1:
        raise ValueError("Require 0 <= start < end <= 1")
    if not math.isfinite(slope_tolerance) or slope_tolerance < 0:
        raise ValueError("slope_tolerance must be finite and nonnegative")
    if interpolation not in ("trilinear", "tetrahedral"):
        raise ValueError("interpolation must be trilinear or tetrahedral")
    values = torch.linspace(start, end, samples, device=tables.device, dtype=tables.dtype)
    ramp = values[None, None, None].expand(tables.shape[0], 3, 1, -1)
    operation = apply_lut if interpolation == "trilinear" else apply_tetrahedral_lut
    output = operation(ramp, tables).squeeze(2).cpu()
    if not bool(torch.isfinite(output).all()):
        raise ValueError("Neutral ramp produced non-finite values")
    slopes = output.diff(dim=-1) / ((end - start) / (samples - 1))
    negative = (slopes < -slope_tolerance).float()
    return {
        "interpolation": interpolation,
        "samples": samples,
        "start": start,
        "end": end,
        "slope_tolerance": slope_tolerance,
        "luts": [
            {
                "index": index,
                "start_rgb": row[:, 0].tolist(),
                "end_rgb": row[:, -1].tolist(),
                "min_slope_rgb": slopes[index].amin(-1).tolist(),
                "max_slope_rgb": slopes[index].amax(-1).tolist(),
                "negative_slope_fraction_rgb": negative[index].mean(-1).tolist(),
                "negative_slope_fraction": float(negative[index].mean()),
            }
            for index, row in enumerate(output)
        ],
    }
