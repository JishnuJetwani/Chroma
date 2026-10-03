"""Reproducible image metrics for encoded sRGB tensors in [0, 1].

SSIM is a Gaussian-window, channel-averaged metric. With a validity mask, only
windows whose complete reflected support is valid are scored.
Delta E is CIE76 in D65 Lab, deliberately not CIEDE2000. PSNR is capped at
120 dB (MSE floor 1e-12) to keep perfect-match reports JSON-compatible.
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _validate_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.ndim != 4 or a.shape != b.shape or a.shape[1] != 3:
        raise ValueError("Expected matching BCHW RGB tensors")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Image metrics require finite values")


def _mask(mask: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(image[:, :1])
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[0] != image.shape[0] or mask.shape[2:] != image.shape[2:]:
        raise ValueError("Mask must have shape B1HW (or BHW)")
    if mask.shape[1] != 1:
        raise ValueError("Use one spatial mask channel")
    mask = mask.to(device=image.device, dtype=image.dtype)
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError("Mask weights must be finite and in [0, 1]")
    return mask


def masked_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over channels and spatial axes; empty masks return NaN per image."""
    weights = _mask(mask, value)
    denominator = weights.sum((1, 2, 3)) * value.shape[1]
    result = (value * weights).sum((1, 2, 3)) / denominator.clamp_min(1e-12)
    return torch.where(denominator > 0, result, torch.full_like(result, float("nan")))


def psnr(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    _validate_pair(prediction, target)
    mse = masked_mean((prediction.float() - target.float()).square(), mask)
    return -10 * torch.log10(mse.clamp_min(1e-12))


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """SSIM with population covariance, reflect padding, and data range one.

    Masked protocol v2 excludes every center whose window touches a pixel with
    validity below 1-1e-6. Padding and privacy pixels cannot affect retained
    moments. An image with no fully valid windows returns NaN, not a zero score.
    """
    _validate_pair(prediction, target)
    if window_size < 1 or sigma <= 0:
        raise ValueError("window_size and sigma must be positive")
    a, b = prediction.float(), target.float()
    size = min(window_size, a.shape[-2], a.shape[-1])
    size -= (size + 1) % 2
    coordinates = torch.arange(size, dtype=a.dtype, device=a.device) - size // 2
    kernel = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, size, size)

    def blur(value: torch.Tensor) -> torch.Tensor:
        pad = size // 2
        if pad:
            value = F.pad(value, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(value, kernel, groups=3)

    mean_a, mean_b = blur(a), blur(b)
    variance_a = (blur(a.square()) - mean_a.square()).clamp_min(0)
    variance_b = (blur(b.square()) - mean_b.square()).clamp_min(0)
    covariance = blur(a * b) - mean_a * mean_b
    numerator = (2 * mean_a * mean_b + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_a.square() + mean_b.square() + 0.01**2) * (variance_a + variance_b + 0.03**2)
    if mask is not None:
        validity = _mask(mask, a)
        invalid = (validity < 1 - 1e-6).to(a.dtype)
        pad = size // 2
        if pad:
            invalid = F.pad(invalid, (pad, pad, pad, pad), mode="reflect")
        valid_support = 1 - F.max_pool2d(invalid, size, stride=1)
        mask = validity * valid_support
    return masked_mean(numerator / denominator, mask)


def srgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """Encoded sRGB -> CIE Lab with the sRGB D65 reference white."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected BCHW RGB tensor")
    rgb = image.float().clamp(0, 1)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    matrix = linear.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.einsum("ij,bjhw->bihw", matrix, linear)
    xyz = xyz / linear.new_tensor([0.95047, 1.0, 1.08883])[None, :, None, None]
    delta = 6 / 29
    transformed = torch.where(xyz > delta**3, xyz.clamp_min(1e-12).pow(1 / 3), xyz / (3 * delta**2) + 4 / 29)
    x, y, z = transformed.unbind(1)
    return torch.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), dim=1)


def delta_e76(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    _validate_pair(prediction, target)
    distances = (srgb_to_lab(prediction) - srgb_to_lab(target)).square().sum(1, keepdim=True).sqrt()
    return masked_mean(distances, mask)


def per_image_metrics(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    _validate_pair(prediction, target)
    difference = prediction.float() - target.float()
    return {
        "mae": masked_mean(difference.abs(), mask),
        "mse": masked_mean(difference.square(), mask),
        "psnr": psnr(prediction, target, mask),
        "ssim": ssim(prediction, target, mask),
        "delta_e76": delta_e76(prediction, target, mask),
    }


def clipping_metrics(
    image: torch.Tensor, reference: torch.Tensor | None = None, mask: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Fractions of pixels with ANY clipped channel; these are diagnostics, not fault labels."""
    low = (image <= 0).any(1, keepdim=True).float()
    high = (image >= 1).any(1, keepdim=True).float()
    result = {
        "clipped_low_fraction": masked_mean(low, mask),
        "clipped_high_fraction": masked_mean(high, mask),
        "all_channels_white_fraction": masked_mean((image >= 1).all(1, keepdim=True).float(), mask),
        "out_of_range_fraction": masked_mean(((image < 0) | (image > 1)).any(1, keepdim=True).float(), mask),
    }
    if reference is not None:
        _validate_pair(image, reference)
        result["new_clipped_high_fraction"] = masked_mean(
            high * (~(reference >= 1).any(1, keepdim=True)).float(), mask
        )
        result["new_clipped_low_fraction"] = masked_mean(
            low * (~(reference <= 0).any(1, keepdim=True)).float(), mask
        )
    return result


def locality_metrics(
    clean_graded: torch.Tensor,
    corrupt_graded: torch.Tensor,
    corrupted_input: torch.Tensor,
    clean_target: torch.Tensor,
    mask: torch.Tensor,
    boundary_radius: int = 5,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Recovery vs clean reference; locality compares two SAME-STYLE model outputs.

    A positive recovery_mae_gain means improvement over the corrupted input.
    Boundary drift is a proxy for artifact inspection, not a validated halo score.
    """
    _validate_pair(clean_graded, corrupt_graded)
    _validate_pair(corrupted_input, clean_target)
    valid = _mask(valid_mask, clean_graded)
    weights = _mask(mask, clean_graded) * valid
    if boundary_radius < 0:
        raise ValueError("boundary_radius must be non-negative")
    support = (weights > 0.01).float()
    radius = boundary_radius
    if radius:
        size = 2 * radius + 1
        dilated = F.max_pool2d(support, size, stride=1, padding=radius)
        eroded = -F.max_pool2d(-support, size, stride=1, padding=radius)
    else:
        dilated = eroded = support
    initial = masked_mean((corrupted_input - clean_target).abs(), weights)
    corrected = masked_mean((corrupt_graded - clean_target).abs(), weights)
    clean_error = masked_mean((clean_graded - clean_target).abs(), weights)
    drift = (clean_graded - corrupt_graded).abs()
    return {
        "region_input_mae": initial,
        "region_output_mae": corrected,
        "clean_region_output_mae": clean_error,
        "corruption_penalty_mae": corrected - clean_error,
        "recovery_mae_gain": initial - corrected,
        "counterfactual_region_drift_mae": masked_mean(drift, weights),
        "outside_drift_mae": masked_mean(drift, (1 - dilated) * valid),
        "boundary_drift_mae": masked_mean(drift, (dilated - eroded).clamp(0, 1) * valid),
    }


def noop_metrics(
    output: torch.Tensor,
    input_image: torch.Tensor,
    mask: torch.Tensor | None = None,
    threshold: float = 2 / 255,
) -> dict[str, torch.Tensor]:
    """Use on explicitly accepted/no-op examples, not arbitrary unedited photographs."""
    _validate_pair(output, input_image)
    difference = (output - input_image).abs()
    return {
        "noop_mae": masked_mean(difference, mask),
        "noop_changed_fraction": masked_mean((difference.amax(1, keepdim=True) > threshold).float(), mask),
    }


def scene_bootstrap(
    values: Sequence[float],
    scene_ids: Sequence[str],
    seed: int = 0,
    samples: int = 2000,
    confidence: float = 0.95,
) -> dict:
    """Equal-weight scene means, resampled by scene (never pixels or repeated votes)."""
    if len(values) != len(scene_ids) or samples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid values, scene IDs, samples or confidence")
    grouped = {}
    for value, scene in zip(values, scene_ids, strict=False):
        if np.isfinite(value):
            grouped.setdefault(str(scene), []).append(float(value))
    if not grouped:
        return {"mean": None, "ci_low": None, "ci_high": None, "scenes": 0, "observations": 0}
    means = np.asarray([np.mean(v) for v in grouped.values()], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        estimates[index] = rng.choice(means, size=len(means), replace=True).mean()
    tail = (1 - confidence) / 2
    low, high = np.quantile(estimates, [tail, 1 - tail])
    return {
        "mean": float(means.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "scenes": len(grouped),
        "observations": sum(map(len, grouped.values())),
        "bootstrap_samples": samples,
        "confidence": confidence,
        "seed": seed,
    }
