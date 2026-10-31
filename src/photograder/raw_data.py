"""Precision-preserving expert targets and aligned floating-point RAW pairs."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

RENDER_VERSION = "fivek-libraw-common-scale-dngcrop-lcms16-float-target-v3"


def _lcms():
    name = ctypes.util.find_library("lcms2")
    if not name:
        raise RuntimeError("LittleCMS2 shared library is required for 16-bit ICC targets")
    lib = ctypes.CDLL(name)
    signatures = {
        "cmsOpenProfileFromMem": ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_void_p),
        "cmsCreate_sRGBProfile": ([], ctypes.c_void_p),
        "cmsGetColorSpace": ([ctypes.c_void_p], ctypes.c_uint32),
        "cmsCreateTransform": (
            [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
            ],
            ctypes.c_void_p,
        ),
        "cmsDoTransform": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32], None),
        "cmsDeleteTransform": ([ctypes.c_void_p], None),
        "cmsCloseProfile": ([ctypes.c_void_p], None),
        "cmsGetEncodedCMMversion": ([], ctypes.c_uint32),
    }
    for function, (arguments, result) in signatures.items():
        getattr(lib, function).argtypes = arguments
        getattr(lib, function).restype = result
    return lib


def icc_rgb16_to_srgb_float(pixels: np.ndarray, icc: bytes) -> tuple[np.ndarray, dict]:
    """LittleCMS RGB16 input -> float sRGB; no intermediate 8-bit conversion."""
    if pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Expected HWC uint16 RGB expert image")
    if not icc:
        raise ValueError("Expert TIFF requires its embedded ICC profile")
    lib = _lcms()
    buffer = ctypes.create_string_buffer(icc)
    source_profile = lib.cmsOpenProfileFromMem(buffer, len(icc))
    destination_profile = lib.cmsCreate_sRGBProfile()
    transform = None
    try:
        if not source_profile or not destination_profile:
            raise ValueError("Could not open ICC profiles")
        if lib.cmsGetColorSpace(source_profile) != 0x52474220:
            raise ValueError("Expected an RGB source ICC profile")
        rgb16 = (4 << 16) | (3 << 3) | 2
        rgb_float = (1 << 22) | (4 << 16) | (3 << 3) | 4
        transform = lib.cmsCreateTransform(
            source_profile, rgb16, destination_profile, rgb_float, 1, 0x0100 | 0x0040
        )
        if not transform:
            raise RuntimeError("LittleCMS could not construct the 16-bit to float transform")
        source = np.ascontiguousarray(pixels)
        output = np.empty(source.shape, dtype=np.float32)
        lib.cmsDoTransform(
            transform, source.ctypes.data, output.ctypes.data, source.shape[0] * source.shape[1]
        )
        if not np.isfinite(output).all():
            raise ValueError("ICC transform produced nonfinite target pixels")
        metadata = {
            "engine": "LittleCMS2",
            "version": lib.cmsGetEncodedCMMversion(),
            "source_bits": 16,
            "target_storage": "float32",
            "intent": "relative_colorimetric",
            "icc_sha256": hashlib.sha256(icc).hexdigest(),
            "pre_sdr_clip_min": float(output.min()),
            "pre_sdr_clip_max": float(output.max()),
        }
        return output.clip(0, 1), metadata
    finally:
        if transform:
            lib.cmsDeleteTransform(transform)
        if source_profile:
            lib.cmsCloseProfile(source_profile)
        if destination_profile:
            lib.cmsCloseProfile(destination_profile)


def orient_image(image: np.ndarray, orientation: int) -> np.ndarray:
    """Apply TIFF/EXIF orientation exactly once to stored pixel coordinates."""
    operations = {
        1: lambda a: a,
        2: lambda a: a[:, ::-1],
        3: lambda a: a[::-1, ::-1],
        4: lambda a: a[::-1],
        5: lambda a: a.transpose(1, 0, 2),
        6: lambda a: np.rot90(a, -1),
        7: lambda a: a.transpose(1, 0, 2)[::-1, ::-1],
        8: lambda a: np.rot90(a, 1),
    }
    if orientation not in operations:
        raise ValueError(f"Unsupported TIFF orientation {orientation}")
    return np.ascontiguousarray(operations[orientation](image))


def load_expert_tiff(path_or_buffer) -> tuple[np.ndarray, dict]:
    import tifffile

    if isinstance(path_or_buffer, bytes):
        path_or_buffer = io.BytesIO(path_or_buffer)
    with tifffile.TiffFile(path_or_buffer) as tif:
        page = tif.pages[0]
        image = page.asarray()
        if image.dtype != np.uint16 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Expected original 16-bit RGB TIFF")
        profile = page.tags.get(34675)
        orientation = int(page.tags[274].value) if 274 in page.tags else 1
        converted, metadata = icc_rgb16_to_srgb_float(image, bytes(profile.value) if profile else b"")
        metadata.update({"stored_dimensions": list(image.shape[1::-1]), "orientation": orientation})
        return orient_image(converted, orientation), metadata


def aligned_pair(
    raw: np.ndarray, target: np.ndarray | None, max_border=64, *, dng_geometry=None, decoder_metadata=None
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Apply exact sensor crop and EXIF orientation before checking target size.

    DNG DefaultCropOrigin is relative to ActiveArea. LibRaw's unoriented output
    may already exclude left/top sensor margins; subtract those exactly once.
    A bounded center fallback exists only when default crop metadata is absent.
    """
    for name, image in (("RAW", raw), ("target", target)):
        if name == "target" and image is None:
            continue
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError(f"Expected nonempty HWC RGB {name} image")
        if not np.isfinite(image).all():
            raise ValueError(f"Nonfinite {name} image")
    rh, rw = raw.shape[:2]
    th, tw = target.shape[:2] if target is not None else (None, None)
    audit = {"raw_dimensions_before_orientation": [rw, rh]}
    if target is not None:
        audit["target_dimensions"] = [tw, th]
    if dng_geometry is not None:
        if decoder_metadata is None or decoder_metadata.get("orientation_applied") is not False:
            raise ValueError("DNG crop requires a verified unoriented RAW decode")
        geometry = decoder_metadata.get("original_geometry", decoder_metadata.get("sizes"))
        if not isinstance(geometry, dict):
            raise ValueError("Missing LibRaw visible sensor geometry")

        def integers(values, length, description):
            try:
                numbers = [float(x) for x in values]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Missing or invalid {description}") from error
            if len(numbers) != length or any(not np.isfinite(x) or not x.is_integer() for x in numbers):
                raise ValueError(f"Noninteger or invalid {description}")
            return [int(x) for x in numbers]

        sensor_w, sensor_h = integers(dng_geometry.get("fullsensor_size_wh"), 2, "DNG sensor dimensions")
        decoded_sensor = integers(
            [geometry.get("raw_width"), geometry.get("raw_height")], 2, "LibRaw sensor dimensions"
        )
        if decoded_sensor != [sensor_w, sensor_h]:
            raise ValueError("DNG and LibRaw full sensor dimensions differ")
        left_margin, top_margin = integers(
            [geometry.get("left_margin"), geometry.get("top_margin")], 2, "LibRaw visible sensor margins"
        )
        visible_w, visible_h = integers(
            [geometry.get("width"), geometry.get("height")], 2, "LibRaw visible image dimensions"
        )
        if (
            left_margin < 0
            or top_margin < 0
            or [visible_w, visible_h] != [rw, rh]
            or left_margin + rw > sensor_w
            or top_margin + rh > sensor_h
        ):
            raise ValueError("Unoriented decoder output does not match its visible sensor rectangle")
        if dng_geometry.get("default_scale_xy", [1, 1]) != [1, 1]:
            raise ValueError("Nonunit DNG DefaultScale requires explicit resampling")
        orientation = dng_geometry.get("exif_orientation")
        if not isinstance(orientation, int) or orientation not in range(1, 9):
            raise ValueError("Invalid DNG EXIF orientation")
        origin = dng_geometry.get("default_crop_origin_xy")
        crop_size = dng_geometry.get("default_crop_size_wh")
        if (origin is None) != (crop_size is None):
            raise ValueError("Incomplete DNG default crop metadata")
        audit.update(
            {
                "dng_crop_metadata": dng_geometry,
                "decoder_visible_sensor_rectangle_xyxy": [
                    left_margin,
                    top_margin,
                    left_margin + rw,
                    top_margin + rh,
                ],
                "orientation_applied": orientation,
            }
        )
        if origin is not None:
            active_top, active_left, active_bottom, active_right = integers(
                dng_geometry.get("active_area_yxyx"), 4, "DNG ActiveArea"
            )
            origin_x, origin_y = integers(origin, 2, "DNG DefaultCropOrigin")
            crop_w, crop_h = integers(crop_size, 2, "DNG DefaultCropSize")
            if (
                not 0 <= active_left < active_right <= sensor_w
                or not 0 <= active_top < active_bottom <= sensor_h
                or min(origin_x, origin_y) < 0
                or min(crop_w, crop_h) < 1
                or origin_x + crop_w > active_right - active_left
                or origin_y + crop_h > active_bottom - active_top
            ):
                raise ValueError("DNG default crop lies outside its active sensor area")
            sensor_x, sensor_y = active_left + origin_x, active_top + origin_y
            left, top = sensor_x - left_margin, sensor_y - top_margin
            if left < 0 or top < 0 or left + crop_w > rw or top + crop_h > rh:
                raise ValueError("DNG default crop lies outside LibRaw visible pixels")
            cropped = orient_image(raw[top : top + crop_h, left : left + crop_w], orientation)
            if target is not None and cropped.shape[:2] != target.shape[:2]:
                raise ValueError(
                    f"Exact DNG crop geometry {cropped.shape[1]}x{cropped.shape[0]} "
                    f"differs from target {tw}x{th}; refusing crop or stretch"
                )
            audit.update(
                {
                    "crop_method": "DNG ActiveArea + DefaultCropOrigin/Size",
                    "raw_sensor_crop_xyxy": [sensor_x, sensor_y, sensor_x + crop_w, sensor_y + crop_h],
                    "raw_visible_crop_xyxy": [left, top, left + crop_w, top + crop_h],
                    "raw_crop_dimensions_before_orientation": [crop_w, crop_h],
                    "raw_dimensions_after_orientation": [cropped.shape[1], cropped.shape[0]],
                    "registration": "Exact metadata crop and orientation; no subpixel registration claim",
                }
            )
            return cropped, target, audit
        raw = orient_image(raw, orientation)
        rh, rw = raw.shape[:2]
    if target is None:
        audit.update(
            {
                "crop_method": "full visible area; DNG default crop absent",
                "raw_dimensions_after_orientation": [rw, rh],
            }
        )
        return raw, None, audit
    # A missing default crop is distinguishable from invalid/unsupported crop
    # metadata. Invalid metadata above must never silently reach this fallback.
    dh, dw = rh - th, rw - tw
    if dh < 0 or dw < 0 or dh > max_border or dw > max_border:
        raise ValueError(f"Unverified pair geometry: RAW {rw}x{rh}, target {tw}x{th}")
    top, left = dh // 2, dw // 2
    cropped = raw[top : top + th, left : left + tw]
    audit.update(
        {
            "crop_method": "bounded center fallback; DNG default crop absent",
            "raw_dimensions_after_orientation": [rw, rh],
            "raw_center_crop": [left, top, left + tw, top + th],
            "registration": "Approximate center crop; not a subpixel registration guarantee",
        }
    )
    return np.ascontiguousarray(cropped), target, audit


def crop_dng_raw(raw: np.ndarray, *, dng_geometry: dict, decoder_metadata: dict) -> tuple[np.ndarray, dict]:
    """Apply the training pipeline's exact DNG crop/orientation at inference.

    No target or dummy native image is allocated. If default crop tags are
    absent, retain the whole visible image and apply only its EXIF orientation.
    Invalid metadata still fails through the same strict checks as training.
    """
    if not isinstance(dng_geometry, dict):
        raise TypeError("DNG geometry metadata is required")
    cropped, _, audit = aligned_pair(raw, None, dng_geometry=dng_geometry, decoder_metadata=decoder_metadata)
    return cropped, audit


def padded_tensor(image: np.ndarray, size: int) -> tuple[torch.Tensor, torch.Tensor, list]:
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None]
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    tensor = F.interpolate(tensor, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    top, left = (size - nh) // 2, (size - nw) // 2
    tensor = F.pad(tensor, (left, size - nw - left, top, size - nh - top), mode="replicate")[0]
    mask = torch.zeros(1, size, size)
    mask[:, top : top + nh, left : left + nw] = 1
    return tensor, mask, [left, top, left + nw, top + nh]


def prepare_raw_pair(input_path, target_path, destination, *, scene_id, split, size=256):
    from .raw_acquisition import dng_crop_metadata
    from .raw_io import decode_raw

    crop = dng_crop_metadata(input_path)
    if hasattr(input_path, "seek"):
        input_path.seek(0)
    raw = decode_raw(input_path, apply_orientation=False)
    target, color = load_expert_tiff(target_path)
    source, target, geometry = aligned_pair(
        raw.linear_rgb, target, dng_geometry=crop, decoder_metadata=raw.metadata
    )
    x, mask, box = padded_tensor(source, size)
    y, target_mask, target_box = padded_tensor(target, size)
    if target_box != box or not torch.equal(mask, target_mask):
        raise ValueError("Prepared geometry mismatch")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scene_id": scene_id,
        "split": split,
        "style_id": 0,
        "render_version": RENDER_VERSION,
        "raw": raw.metadata,
        "target": color,
        "geometry": geometry,
        "content_box": box,
        "training_size": size,
        "input_min": float(x.min()),
        "input_max": float(x.max()),
        "input_above_one_fraction": float(((x > 1).any(0, keepdim=True) * mask).sum() / mask.sum()),
    }
    temporary = destination.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            image=x.numpy(),
            target=y.numpy(),
            valid_mask=mask.numpy(),
            metadata=np.array(json.dumps(metadata)),
        )
    temporary.replace(destination)
    return {
        **metadata,
        "path": destination.name,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
    }


class RawPairDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split):
        self.manifest = Path(manifest)
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        self.rows = [row for row in self.rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No RAW pairs in {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.manifest.parent / row["path"], allow_pickle=False) as sample:
            return {key: torch.from_numpy(sample[key].copy()) for key in ("image", "target", "valid_mask")}
