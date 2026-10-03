"""Deterministic, recoverable local faults with exact finite support.

These are synthetic training perturbations, not examples of human preference.
The edge_aware family is excluded from the default training families so it can
serve as a held-out corruption generator. All random draws use a local CPU RNG.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FAMILIES = ("ellipse", "gradient", "irregular", "edge_aware")
CORRUPTION_VERSION = "local-recoverable-v2-headroom"


def _srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(0).pow(2.4))


def _linear_to_srgb(x):
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12).pow(1 / 2.4) - 0.055)


def _rand(generator, low=0.0, high=1.0):
    return low + (high - low) * torch.rand((), generator=generator).item()


def region_mask(image, family, generator):
    """Generate a soft [1,H,W] mask whose complement is exactly zero."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown corruption family {family}; expected {FAMILIES}")
    _, h, w = image.shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    cx, cy = _rand(generator, -0.5, 0.5), _rand(generator, -0.5, 0.5)
    rx, ry = _rand(generator, 0.35, 0.95), _rand(generator, 0.35, 0.95)
    distance = (((x - cx) / rx).square() + ((y - cy) / ry).square()).sqrt()
    ellipse = ((1 - distance) / 0.3).clamp(0, 1)
    ellipse = ellipse.square() * (3 - 2 * ellipse)
    if family == "ellipse":
        mask = ellipse
    elif family == "gradient":
        angle = _rand(generator, 0, 2 * math.pi)
        projection = x * math.cos(angle) + y * math.sin(angle)
        position = _rand(generator, -0.5, 0.5)
        mask = (1 - (projection - position).abs() / _rand(generator, 0.3, 0.8)).clamp(0, 1)
        mask = mask.square() * (3 - 2 * mask)
    elif family == "irregular":
        coarse = torch.rand((1, 1, 5, 5), generator=generator)
        noise = F.interpolate(coarse, (h, w), mode="bicubic", align_corners=False)[0, 0]
        mask = ((noise - 0.35) / 0.3).clamp(0, 1) * ellipse
    else:
        luminance = (image.detach().cpu() * torch.tensor([0.2126, 0.7152, 0.0722])[:, None, None]).sum(0)
        center_y, center_x = int((cy + 1) * (h - 1) / 2), int((cx + 1) * (w - 1) / 2)
        delta = (luminance - luminance[center_y, center_x]).abs()
        mask = ((0.25 - delta) / 0.1).clamp(0, 1) * ellipse
    return mask.unsqueeze(0).to(image.device, image.dtype)


def _apply_fault(image, mask, operation, generator):
    linear = _srgb_to_linear(image)
    if operation == "exposure":
        stops = _rand(generator, 0.25, 0.8) * (-1 if _rand(generator) < 0.5 else 1)
        delta = mask * stops
    else:
        luminance = (linear * linear.new_tensor([0.2126, 0.7152, 0.0722])[:, None, None]).sum(0, keepdim=True)
        log_luma = luminance.clamp_min(1e-6).log2()
        base = F.avg_pool2d(F.pad(log_luma[None], (8, 8, 8, 8), mode="replicate"), 17, stride=1)[0]
        exponent = _rand(generator, 0.25, 0.65) * (-1 if _rand(generator) < 0.5 else 1)
        factor = 2**exponent
        delta = (log_luma - base) * (factor - 1) * mask
        # One existing saturated channel must not collapse an entire regional
        # contrast fault to zero. Limit only its positive luminance shift by
        # local RGB headroom; darker/detail pixels still receive the fault.
        headroom = -linear.amax(dim=0, keepdim=True).clamp(1e-8, 1).log2()
        delta = torch.minimum(delta, headroom * 0.98)
    # Reduce fault amplitude globally until there is no newly clipped channel.
    # For exposure with exact white pixels, brighten is unrecoverable: darken.
    if operation == "exposure" and bool((linear * torch.exp2(delta) > 1 + 1e-7).any()):
        delta = -delta.abs()
    scale = 1.0
    for _ in range(16):
        candidate = linear * torch.exp2(delta * scale)
        if bool((candidate <= 1 + 1e-7).all()):
            break
        scale *= 0.5
    else:
        scale = 0.0
    delta = delta * scale
    candidate = linear * torch.exp2(delta)
    # Tiny transfer-function roundoff only; clipping was prevented above.
    output = _linear_to_srgb(candidate.clamp(0, 1)).clamp(0, 1)
    support = mask > 0
    output = torch.where(support.expand_as(image), output, image)
    return (
        output,
        delta,
        {
            "operation": operation,
            "amplitude_scale": scale,
            "max_abs_applied_stops": float(delta.abs().max().cpu()),
        },
    )


@torch.no_grad()
def corrupt_batch(
    images,
    seed,
    families=("ellipse", "gradient", "irregular"),
    allow_identity=True,
    valid_mask=None,
    allow_combinations=False,
):
    """Apply bounded faults to B3HW [0,1] images without global RNG changes.

    ``mask`` is binary support [B,1,H,W], ``soft_mask`` is feathered support.
    ``known_inverse`` stores exposure-inverse stops only for pure exposure and
    identity samples, marked by ``inverse_known``; contrast/combined samples
    require image supervision. ``operation`` and ``parameters`` are per-example
    lists. When provided, valid_mask excludes padding and privacy regions.
    """
    if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16:
        raise ValueError("Expected B3HW images with dimensions >=16")
    if (
        not images.is_floating_point()
        or not bool(torch.isfinite(images).all())
        or bool(((images < 0) | (images > 1)).any())
    ):
        raise ValueError("Images must be finite floating sRGB in [0,1]")
    if not families or any(f not in FAMILIES for f in families):
        raise ValueError("families must contain supported region generators")
    if valid_mask is not None and valid_mask.shape != (images.shape[0], 1, *images.shape[-2:]):
        raise ValueError("valid_mask must be B1HW")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    outputs, supports, soft_masks, inverse, inverse_known, operations, parameters = [], [], [], [], [], [], []
    for index, image in enumerate(images):
        soft = torch.zeros_like(image[:1])
        accumulated = soft.clone()
        output = image
        params = []
        identity = allow_identity and _rand(generator) < 0.2
        count = 0 if identity else (2 if allow_combinations and _rand(generator) < 0.35 else 1)
        ops = []
        for _ in range(count):
            family = families[int(_rand(generator) * len(families))]
            mask = region_mask(image, family, generator)
            if valid_mask is not None:
                mask = mask * (valid_mask[index] > 0).to(mask.dtype)
            operation = "exposure" if _rand(generator) < 0.5 else "contrast"
            output, delta, details = _apply_fault(output, mask, operation, generator)
            soft = torch.maximum(soft, mask)
            accumulated += delta
            params.append({"family": family, **details})
            ops.append(operation)
        operation = "identity" if identity else "+".join(ops)
        known = operation in {"exposure", "identity"}
        outputs.append(output)
        soft_masks.append(soft)
        supports.append((soft > 0).to(soft.dtype))
        inverse.append(-accumulated if known else torch.zeros_like(accumulated))
        inverse_known.append(known)
        operations.append(operation)
        parameters.append(params)
    return {
        "version": CORRUPTION_VERSION,
        "image": torch.stack(outputs),
        "mask": torch.stack(supports),
        "soft_mask": torch.stack(soft_masks),
        "known_inverse": torch.stack(inverse),
        "inverse_known": torch.tensor(inverse_known, device=images.device),
        "operation": operations,
        "parameters": parameters,
    }
