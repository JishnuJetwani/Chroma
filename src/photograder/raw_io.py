"""Decode RGB CFA RAW files into unclipped, scene-linear D65 sRGB.

LibRaw supplies black-subtracted, unbalanced camera RGB. White balance and the
color matrix are applied here in float32, so values above one and negative
matrix results survive for the scene-to-display renderer. LibRaw's demosaic
itself remains integer and can clip interpolation excursions and sensor blacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

RAW_PIPELINE_VERSION = "raw-camera-float-wb-matrix-v2"
SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)


class RawDecodeError(ValueError):
    """The input cannot be decoded with this pipeline's supported assumptions."""


@dataclass(frozen=True)
class RawImage:
    linear_rgb: np.ndarray
    metadata: dict


def _rgb_vector(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape not in ((3,), (4,)) or not np.isfinite(array).all():
        raise RawDecodeError(f"{name} must contain three or four finite channel values.")
    return array[:3]


def _camera_matrix(rgb_xyz_matrix):
    """Reproduce LibRaw cam_xyz_coeff's row normalization and inverse.

    Despite rawpy's property name, cam_xyz maps XYZ to camera, not vice versa.
    Its first three rows are valid here only for ordinary three-color sensors.
    """
    matrix = np.asarray(rgb_xyz_matrix, dtype=np.float64)
    if matrix.shape not in ((3, 3), (4, 3)) or not np.isfinite(matrix).all():
        raise RawDecodeError("Missing or invalid camera XYZ matrix.")
    if matrix.shape == (4, 3) and not np.allclose(matrix[3], 0, atol=1e-8, rtol=0):
        raise RawDecodeError("Four-color camera matrices are not supported.")
    camera_from_srgb = matrix[:3] @ SRGB_TO_XYZ
    sums = camera_from_srgb.sum(axis=1, keepdims=True)
    if np.any(sums <= 1e-6):
        raise RawDecodeError("Missing or invalid camera XYZ matrix white point.")
    normalized = camera_from_srgb / sums
    if np.linalg.cond(normalized) > 1e6:
        raise RawDecodeError("Camera XYZ matrix is singular or ill-conditioned.")
    return np.linalg.inv(normalized).astype(np.float32)


def camera_to_linear_srgb(
    camera_rgb,
    *,
    black_levels,
    white_levels,
    white_balance,
    rgb_xyz_matrix,
    black_subtracted=True,
    normalization_scale=None,
):
    """Normalize camera RGB and apply WB/CCM without clipping.

    ``black_subtracted=True`` is required for the output of ``decode_raw``'s
    LibRaw configuration. Set it to False only for uncorrected camera samples.
    A common scalar preserves sensor-channel ratios, matching LibRaw's
    scale_colors convention. Channel saturation thresholds are not independent
    sensitivity calibrations. The decoder supplies its scalar white minus the
    common black component; standalone calls default to max(white)-min(black).
    """
    image = np.asarray(camera_rgb)
    if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
        raise RawDecodeError("Expected finite HWC camera RGB with three channels.")
    black = _rgb_vector(black_levels, "Black levels")
    white = _rgb_vector(white_levels, "White levels")
    wb = _rgb_vector(white_balance, "White balance")
    if np.any(black < 0) or np.any(white <= black):
        raise RawDecodeError("RAW white levels must exceed nonnegative black levels.")
    if np.any(wb <= 0):
        raise RawDecodeError("As-shot white balance is missing or invalid.")
    matrix = _camera_matrix(rgb_xyz_matrix)
    normalized = image.astype(np.float32, copy=True)
    if not black_subtracted:
        normalized -= black.astype(np.float32)
    if normalization_scale is None:
        normalization_scale = float(np.max(white_levels) - np.min(black_levels))
    if (
        not np.isscalar(normalization_scale)
        or not np.isfinite(normalization_scale)
        or normalization_scale <= 0
    ):
        raise RawDecodeError("RAW normalization scale must be a finite positive scalar.")
    normalized /= np.float32(normalization_scale)
    normalized *= (wb / wb[1]).astype(np.float32)
    return np.ascontiguousarray(normalized @ matrix.T, dtype=np.float32)


def _sensor_clipping(raw, pattern, white):
    """Count saturation in the visible CFA in bounded row chunks."""
    sensor = raw.raw_image_visible
    height, width = sensor.shape
    sizes = raw.sizes
    cols = (np.arange(width) + sizes.left_margin) % pattern.shape[1]
    count = 0
    for top in range(0, height, 256):
        rows = (np.arange(top, min(height, top + 256)) + sizes.top_margin) % pattern.shape[0]
        colors = pattern[rows[:, None], cols[None, :]]
        count += int(np.count_nonzero(sensor[top : top + 256] >= white[colors]))
    return count / sensor.size


def decode_raw(
    path_or_binary_file: str | os.PathLike | BinaryIO, *, apply_orientation=True, max_pixels=60_000_000
) -> RawImage:
    """Decode a local path or binary file object; retain scene highlights.

    Only flat RGB Bayer/X-Trans CFA data with a usable LibRaw cam_xyz matrix
    are accepted. A file extension or camera-list entry alone is not proof of
    support. No thumbnail fallback, color-matrix guess, or output clamp occurs.
    """
    try:
        import rawpy
    except ImportError as error:
        raise RawDecodeError("RAW support requires rawpy; install the project's RAW dependencies.") from error
    if not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer.")
    raw = rawpy.RawPy()
    try:
        if isinstance(path_or_binary_file, (str, os.PathLike)):
            raw.open_file(os.fspath(path_or_binary_file))
        elif hasattr(path_or_binary_file, "read"):
            raw.open_buffer(path_or_binary_file)
        else:
            raise TypeError("RAW input must be a filesystem path or binary file object.")
        sizes = raw.sizes
        if sizes.raw_height * sizes.raw_width > max_pixels:
            raise RawDecodeError(f"RAW sensor exceeds the {max_pixels:,}-pixel decoding limit.")
        if raw.num_colors != 3 or not raw.color_desc.startswith(b"RGB"):
            raise RawDecodeError("Only three-color RGB CFA sensors are supported.")
        raw.unpack()
        sizes = raw.sizes
        if sizes.raw_height * sizes.raw_width > max_pixels:
            raise RawDecodeError(f"Unpacked RAW exceeds the {max_pixels:,}-pixel decoding limit.")
        if raw.raw_type != rawpy.RawType.Flat or raw.raw_image_visible.ndim != 2:
            raise RawDecodeError("Only flat RGB CFA RAW files are supported.")
        pattern = raw.raw_pattern
        if pattern is None or pattern.shape not in ((2, 2), (4, 4), (6, 6)):
            raise RawDecodeError("Only Bayer and X-Trans CFA patterns are supported.")
        if np.any(pattern > 3):
            raise RawDecodeError("Unsupported RAW color-filter pattern.")
        black = np.array(raw.black_level_per_channel, dtype=np.float64)
        specific_white = raw.camera_white_level_per_channel
        white = np.array(
            specific_white if specific_white is not None else [raw.white_level] * 4, dtype=np.float64
        )
        wb = np.array(raw.camera_whitebalance, dtype=np.float64)
        matrix_xyz = raw.rgb_xyz_matrix.copy()
        matrix = _camera_matrix(matrix_xyz)
        # LibRaw subtracts each site's black before merging greens. Unequal
        # blacks are safe with a common normalization scale. Unequal gains are
        # not recoverable after merging, so they still require another path.
        if np.any(pattern == 3):
            if not raw.color_desc.startswith(b"RGBG"):
                raise RawDecodeError("The fourth CFA channel must be a matching green channel.")
            if wb[3] > 0 and not np.isclose(wb[1], wb[3], rtol=1e-5, atol=1e-7):
                raise RawDecodeError("Unequal green-channel white balance requires a four-channel decoder.")
        if np.any(white <= black) or np.any(black < 0) or np.any(wb[:3] <= 0):
            raise RawDecodeError("RAW black/white levels or as-shot white balance are invalid.")
        scalar_white = float(raw.white_level)
        common_black = float(black.min())
        normalization_scale = scalar_white - common_black
        clipping = _sensor_clipping(raw, pattern, white)
        original_geometry = dict(sizes._asdict())
        params = {
            "output_color": rawpy.ColorSpace.raw,
            "output_bps": 16,
            "gamma": (1, 1),
            "no_auto_bright": True,
            "no_auto_scale": True,
            "adjust_maximum_thr": 0.0,
            "bright": 1.0,
            "use_camera_wb": False,
            "use_auto_wb": False,
            "user_wb": [1.0] * 4,
            "highlight_mode": rawpy.HighlightMode.Ignore,
            "demosaic_algorithm": rawpy.DemosaicAlgorithm.AHD,
            "half_size": False,
            "four_color_rgb": False,
            "median_filter_passes": 0,
        }
        if not apply_orientation:
            params["user_flip"] = 0
        camera = raw.postprocess(**params)
        if camera.shape[0] * camera.shape[1] > max_pixels:
            raise RawDecodeError("Demosaiced RAW exceeds the output pixel limit.")
        linear = camera_to_linear_srgb(
            camera,
            black_levels=black,
            white_levels=white,
            white_balance=wb,
            rgb_xyz_matrix=matrix_xyz,
            black_subtracted=True,
            normalization_scale=normalization_scale,
        )
        metadata = {
            "pipeline_version": RAW_PIPELINE_VERSION,
            "rawpy_version": rawpy.__version__,
            "libraw_version": list(rawpy.libraw_version),
            "working_space": "scene-linear sRGB / D65; unclipped float32",
            "matrix_source": "LibRaw cam_xyz; normalized inverse of cam_xyz @ sRGB_to_XYZ",
            "camera_xyz_matrix": matrix_xyz.tolist(),
            "camera_to_srgb_matrix": matrix.tolist(),
            "black_levels": black.tolist(),
            "white_levels": white.tolist(),
            "white_level_source": "camera per-channel" if specific_white is not None else "LibRaw scalar",
            "normalization": "LibRaw-subtracted camera RGB / (LibRaw scalar white - common black); no second subtraction; one scale for RGB",
            "normalization_scale": normalization_scale,
            "normalization_scalar_white": scalar_white,
            "normalization_common_black": common_black,
            "white_balance": wb.tolist(),
            "white_balance_normalized_rgb": (wb[:3] / wb[1]).tolist(),
            "sensor_clip_fraction": clipping,
            "sensor_clip_definition": "visible CFA photosites at or above metadata white level",
            "original_geometry": original_geometry,
            "sizes": original_geometry,
            "source_flip": int(sizes.flip),
            "orientation_applied": bool(apply_orientation),
            "output_geometry": {"height": linear.shape[0], "width": linear.shape[1]},
            "cfa_pattern": pattern.tolist(),
            "color_desc": raw.color_desc.decode("ascii", errors="replace"),
            "decoder_output_dtype": str(camera.dtype),
            "limitations": "Integer LibRaw demosaic and black subtraction can clip interpolation excursions and below-black noise; no sensor highlight reconstruction.",
        }
        return RawImage(linear_rgb=linear, metadata=metadata)
    except rawpy.LibRawError as error:
        raise RawDecodeError(f"LibRaw could not decode this RAW file: {error}") from error
    finally:
        raw.close()
