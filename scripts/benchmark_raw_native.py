#!/usr/bin/env python3
"""Benchmark public FiveK a0007 RAW development on Mac MPS without staging inputs.

Only the official DNG is downloaded, into a bounded in-memory buffer. LibRaw
really decodes that buffer. RawGrader's filesystem decoder and DNG-tag reader
are then substituted with those already-read results. Exact DNG crop, preview
preparation, trained model analysis, cached rendering, and native TIFF export
remain the real production implementations. This is NOT filesystem upload QA.

The generated native TIFF lives in an owned TemporaryDirectory and is deleted
after inspection. Only small before/after PNGs and a JSON report are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image

from photograder.inference import tensor_image
from photograder.raw_acquisition import INDEX_LIMIT, dng_crop_metadata, parse_official_index
from photograder.raw_inference import RawGrader
from photograder.raw_io import RAW_PIPELINE_VERSION, decode_raw
from photograder.raw_model import baseline_raw
from photograder.runtime import atomic_json, file_sha256, synchronize

MAX_DNG_BYTES = 20 * 1024**2
SCENE_ID = "fivek:a0007"


def _download_memory(url: str) -> tuple[io.BytesIO, dict]:
    """Return at most 20 MiB, with no source-file writes or disk-backed spool."""
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "data.csail.mit.edu"
        or not parsed.path.startswith("/graphics/fivek/img/dng/a0007-")
        or not parsed.path.endswith(".dng")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Expected the exact public a0007 DNG URL from the official index")
    started = time.perf_counter()
    buffer = io.BytesIO()
    digest = hashlib.sha256()
    received = 0
    with requests.get(
        url, stream=True, timeout=(15, 60), headers={"Accept-Encoding": "identity"}
    ) as response:
        response.raise_for_status()
        final = urlparse(response.url)
        if final.scheme != "https" or final.netloc != parsed.netloc or final.path != parsed.path:
            raise ValueError("Unexpected redirect away from the official source URL")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > MAX_DNG_BYTES:
            raise ValueError("Official DNG exceeds the fixed 20 MiB in-memory budget")
        for block in response.iter_content(64 * 1024):
            if not block:
                continue
            if received + len(block) > MAX_DNG_BYTES:
                raise ValueError("Official DNG exceeded the fixed 20 MiB in-memory budget while streaming")
            buffer.write(block)
            digest.update(block)
            received += len(block)
        if received == 0:
            raise ValueError("Official DNG response was empty")
        if declared is not None and int(declared) != received:
            raise ValueError("Official DNG response length differs from Content-Length")
    buffer.seek(0)
    return buffer, {
        "url": url,
        "bytes": received,
        "byte_limit": MAX_DNG_BYTES,
        "sha256": digest.hexdigest(),
        "download_seconds": time.perf_counter() - started,
        "storage": "io.BytesIO only; no native source file created",
    }


def _mps_memory() -> dict:
    return {
        "current_tensor_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


def _linear_statistics(image: torch.Tensor) -> dict:
    """Read the CPU image in row chunks instead of making native-sized masks."""
    array = image[0].permute(1, 2, 0).numpy()
    minimum, maximum = float("inf"), float("-inf")
    above_one, negative = 0, 0
    for top in range(0, array.shape[0], 256):
        rows = array[top : top + 256]
        minimum, maximum = min(minimum, float(rows.min())), max(maximum, float(rows.max()))
        above_one += int(np.count_nonzero(np.any(rows > 1, axis=-1)))
        negative += int(np.count_nonzero(np.any(rows < 0, axis=-1)))
    count = array.shape[0] * array.shape[1]
    return {
        "minimum": minimum,
        "maximum": maximum,
        "native_float_bytes": int(array.nbytes),
        "rgb_above_one_pixel_fraction": above_one / count,
        "rgb_negative_pixel_fraction": negative / count,
        "note": "RGB above one is retained headroom, not a sensor clipping classification.",
    }


def _save_small(image: Image.Image, path: Path, edge: int) -> dict:
    preview = image.copy()
    preview.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    preview.save(path, format="PNG")
    return {
        "path": str(path.resolve()),
        "width": preview.width,
        "height": preview.height,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _native_crops(mapped: np.ndarray, analysis, output: Path, device) -> dict:
    """Save three a0007 diagnostics at 1:1 pixels; never resize crop content."""
    # Fixed locations chosen from the retained public-image preview, covering
    # textured sculpture, bright water, and a bright/dark architectural edge.
    regions = {"statue": (1284, 1654), "fountain": (1044, 3284), "wall-shadow": (484, 2034)}
    results = {}
    for name, (left, top) in regions.items():
        if left + 512 > mapped.shape[1] or top + 512 > mapped.shape[0]:
            raise ValueError("Native diagnostic region falls outside the expected a0007 crop")
        source = analysis.linear_rgb[..., top : top + 512, left : left + 512].to(device)
        with torch.inference_mode():
            neutral = tensor_image(baseline_raw(source))
        rendered = np.asarray(mapped[top : top + 512, left : left + 512], dtype=np.float32)
        graded = Image.fromarray(np.rint(rendered / 257).astype(np.uint8))
        pair = Image.new("RGB", (1024, 512))
        pair.paste(neutral, (0, 0))
        pair.paste(graded, (512, 0))
        path = output / f"native-{name}-neutral-left-graded-right.png"
        pair.save(path, format="PNG")
        results[name] = {
            "path": str(path.resolve()),
            "source_crop_xyxy": [left, top, left + 512, top + 512],
            "crop_dimensions_wh": [512, 512],
            "pair_dimensions_wh": [1024, 512],
            "left": "neutral baseline development",
            "right": "trained graded TIFF output",
            "display_scale": "1:1 native pixels; no resizing",
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return results


def benchmark(args) -> dict:
    import tifffile

    if platform.system() != "Darwin" or not torch.backends.mps.is_available():
        raise RuntimeError("This benchmark requires a Mac with Apple MPS; no CPU fallback is used.")
    if not 1 <= args.preview_repeats <= 10:
        raise ValueError("preview_repeats must be between one and ten")
    if not 64 <= args.artifact_size <= 1024 or not 64 <= args.tile_size <= 1024:
        raise ValueError("artifact_size and tile_size must be between 64 and 1024")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained checkpoint is not available yet: {checkpoint}")
    index_path = Path(args.index)
    if index_path.stat().st_size > INDEX_LIMIT:
        raise ValueError("Saved official index exceeds its 8 MiB metadata budget")
    record = parse_official_index(index_path.read_text())[SCENE_ID]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Fail before downloading if the checkpoint is not trained or the RAW
    # normalization version differs from the current v2 decoder.
    started = time.perf_counter()
    grader = RawGrader(checkpoint, device="mps", preview_size=1024, tile_size=args.tile_size)
    synchronize(grader.device)
    model_load_seconds = time.perf_counter() - started
    print(f"Loaded trained {RAW_PIPELINE_VERSION} checkpoint on MPS.", flush=True)

    source_buffer, source = _download_memory(record["input_url"])
    print(f"Downloaded {source['bytes'] / 1024**2:.2f} MiB of public a0007 into RAM only.", flush=True)
    started = time.perf_counter()
    geometry = dng_crop_metadata(source_buffer)
    geometry_seconds = time.perf_counter() - started
    source_buffer.seek(0)
    started = time.perf_counter()
    decoded = decode_raw(source_buffer, apply_orientation=False, max_pixels=60_000_000)
    decode_seconds = time.perf_counter() - started
    source_buffer.close()
    if decoded.metadata["pipeline_version"] != RAW_PIPELINE_VERSION:
        raise ValueError("Decoded photo does not match the current checkpoint pipeline")
    print(f"Real in-memory LibRaw decode completed in {decode_seconds:.3f} s.", flush=True)

    # Warm only generic MPS kernels before the measured photographic analysis.
    started = time.perf_counter()
    with torch.inference_mode():
        warmup = torch.full(
            (1, 3, grader.model.config.preview_size, grader.model.config.preview_size),
            0.18,
            device=grader.device,
        )
        grader.model.analyze(warmup)
        synchronize(grader.device)
        del warmup
    model_warmup_seconds = time.perf_counter() - started
    memory_before_analysis = _mps_memory()
    counts = {
        "filesystem_decoder_substitutions": 0,
        "filesystem_tag_reader_substitutions": 0,
        "photographic_cnn_passes": 0,
    }
    virtual_path = Path("virtual-a0007.dng")

    def cached_decode(path, *, apply_orientation, max_pixels):
        if Path(path) != virtual_path or apply_orientation or max_pixels != 60_000_000:
            raise AssertionError("Unexpected decoder request from RawGrader")
        counts["filesystem_decoder_substitutions"] += 1
        return decoded

    def cached_geometry(path):
        if Path(path) != virtual_path:
            raise AssertionError("Unexpected DNG metadata request from RawGrader")
        counts["filesystem_tag_reader_substitutions"] += 1
        return geometry

    def count_cnn(_module, _inputs):
        counts["photographic_cnn_passes"] += 1

    # A counting hook observes the genuine CNN; it does not replace computation.
    cnn_hook = grader.model.encoder[0].register_forward_pre_hook(count_cnn)
    try:
        with (
            patch("photograder.raw_inference.decode_raw", side_effect=cached_decode),
            patch("photograder.raw_acquisition.dng_crop_metadata", side_effect=cached_geometry),
        ):
            synchronize(grader.device)
            started = time.perf_counter()
            analysis = grader.analyze(virtual_path)
            synchronize(grader.device)
            analysis_seconds = time.perf_counter() - started
            height, width = analysis.source_size
            if not 11_000_000 <= height * width <= 13_500_000:
                raise ValueError(
                    f"a0007 did not yield the expected approximately 12 MP crop: {width}x{height}"
                )
            if analysis.linear_rgb.device.type != "cpu" or analysis.linear_rgb.dtype != torch.float32:
                raise AssertionError("Native RAW cache must remain CPU float32")
            crop_audit = analysis.raw_metadata["development_geometry"]
            if crop_audit["crop_method"] != "DNG ActiveArea + DefaultCropOrigin/Size":
                raise AssertionError("Native benchmark requires the exact DNG metadata crop")
            memory_after_analysis = _mps_memory()
            print(
                f"Exact crop and real model analysis: {width} × {height}, {analysis_seconds:.3f} s.",
                flush=True,
            )

            # Report first photographic preview separately from cached repeats.
            synchronize(grader.device)
            started = time.perf_counter()
            graded_preview = grader.render_preview(analysis)
            synchronize(grader.device)
            first_preview_seconds = time.perf_counter() - started
            preview_seconds = []
            for _ in range(args.preview_repeats):
                synchronize(grader.device)
                started = time.perf_counter()
                graded_preview = grader.render_preview(analysis)
                synchronize(grader.device)
                preview_seconds.append(time.perf_counter() - started)
            artifacts = {
                "before": _save_small(analysis.preview, output / "before.png", args.artifact_size),
                "after": _save_small(graded_preview, output / "after.png", args.artifact_size),
            }
            native_statistics = _linear_statistics(analysis.linear_rgb)
            memory_before_export = _mps_memory()
            with tempfile.TemporaryDirectory(prefix="photograder-native-mps-") as owned_directory:
                native_tiff = Path(owned_directory) / "a0007-developed.tif"
                synchronize(grader.device)
                started = time.perf_counter()
                exported = grader.export(analysis, native_tiff)
                synchronize(grader.device)
                export_wall_seconds = time.perf_counter() - started
                export_bytes = native_tiff.stat().st_size
                with tifffile.TiffFile(native_tiff) as document:
                    page = document.pages[0]
                    if page.shape != (height, width, 3) or page.dtype != np.dtype("uint16"):
                        raise AssertionError(
                            "Native TIFF geometry or precision differs from the rendered source"
                        )
                    profile = page.tags.get(34675)
                    if profile is None or not profile.value:
                        raise AssertionError("Native TIFF is missing its sRGB ICC profile")
                    icc_bytes = len(profile.value)
                # The production exporter writes a contiguous uncompressed TIFF;
                # inspect its existing file through a mapping, not another copy.
                mapped = tifffile.memmap(native_tiff, mode="r")
                try:
                    sample = mapped[::16, ::16]
                    fine_precision_fraction = float(np.mean(sample % 257 != 0))
                    if fine_precision_fraction == 0:
                        raise AssertionError("TIFF samples appear to have been quantized through eight bits")
                    native_inspection = {
                        "shape_hwc": list(mapped.shape),
                        "dtype": str(mapped.dtype),
                        "icc_profile_bytes": icc_bytes,
                        "minimum_code": int(mapped.min()),
                        "maximum_code": int(mapped.max()),
                        "sampled_non_eight_bit_code_fraction": fine_precision_fraction,
                        "bytes": export_bytes,
                        "sha256": file_sha256(native_tiff),
                    }
                    native_crop_artifacts = (
                        _native_crops(mapped, analysis, output, grader.device) if args.native_crops else {}
                    )
                finally:
                    mapped._mmap.close()
                    del mapped
                memory_after_export = _mps_memory()
            if native_tiff.exists() or Path(owned_directory).exists():
                raise AssertionError("Owned native TIFF temporary directory was not removed")
            print(
                f"Native 16-bit TIFF export: {export_wall_seconds:.3f} s; inspected and deleted.", flush=True
            )
    finally:
        cnn_hook.remove()
    if counts != {
        "filesystem_decoder_substitutions": 1,
        "filesystem_tag_reader_substitutions": 1,
        "photographic_cnn_passes": 1,
    }:
        raise AssertionError(f"Cached preview/export unexpectedly reanalyzed or decoded the image: {counts}")

    report = {
        "benchmark": "public FiveK a0007 native RAW development on Mac MPS",
        "passed": True,
        "scene_id": SCENE_ID,
        "source": source,
        "official_index_sha256": file_sha256(index_path),
        "camera_description_from_official_index": record["camera_exif_description"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_metadata": grader.checkpoint.get("metadata", {}),
        "raw_pipeline_version": RAW_PIPELINE_VERSION,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": str(grader.device),
            "rawpy": decoded.metadata.get("rawpy_version"),
            "libraw": decoded.metadata.get("libraw_version"),
        },
        "native_dimensions": {"width": width, "height": height, "megapixels": height * width / 1e6},
        "input_linear_statistics": native_statistics,
        "crop": crop_audit,
        "sensor_clip_fraction": decoded.metadata.get("sensor_clip_fraction"),
        "timings_seconds": {
            "download": source["download_seconds"],
            "dng_tag_read": geometry_seconds,
            "real_libraw_decode_from_memory": decode_seconds,
            "model_load": model_load_seconds,
            "generic_model_warmup": model_warmup_seconds,
            "crop_preview_analysis_and_baseline": analysis_seconds,
            "first_cached_preview": first_preview_seconds,
            "cached_preview_repeats": preview_seconds,
            "cached_preview_median": statistics.median(preview_seconds),
            "native_tiff_export_wall": export_wall_seconds,
            "native_tiff_export_internal": exported["seconds"],
        },
        "tile_size": args.tile_size,
        "preview_long_edge": grader.preview_size,
        "cache_reuse_counts": counts,
        "native_tiff_inspection": native_inspection,
        "native_tiff_deleted": True,
        "native_source_files_created": False,
        "expert_tiff_downloaded": False,
        "artifacts": artifacts,
        "native_crop_artifacts": native_crop_artifacts,
        "mps_memory_snapshots_bytes": {
            "before_analysis": memory_before_analysis,
            "after_analysis": memory_after_analysis,
            "before_export": memory_before_export,
            "after_export": memory_after_export,
        },
        "mps_memory_measurement": "Point-in-time allocation snapshots; these are not peak-memory measurements.",
        "scope": {
            "filesystem_input_loading_tested": False,
            "substitution": "Only filesystem decoder and DNG tag-reader calls were replaced by actual "
            "in-memory decoded pixels and real metadata read from the official DNG. "
            "The virtual .dng path was never created.",
            "real_operations": [
                "bounded public DNG download into BytesIO",
                "LibRaw decode from BytesIO",
                "DNG metadata parse",
                "exact DNG crop and EXIF orientation",
                "native CPU float cache",
                "antialiased analysis preview",
                "trained CNN on MPS",
                "cached preview rendering",
                "native tiled MPS render",
                "16-bit TIFF writing and inspection",
                "owned temporary export cleanup",
            ],
            "quality_claim": "One-photo execution and precision check; no expert target, quality metric, "
            "or generalization claim is derived from this benchmark.",
        },
    }
    atomic_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/raw-pilot-20261030/shared/model.pt")
    parser.add_argument("--index", default="data/raw_fivek_plan/official-index.html")
    parser.add_argument("--output", default="runs/raw-pilot-20261030/native_mps_check")
    parser.add_argument("--preview-repeats", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--artifact-size", type=int, default=768)
    parser.add_argument(
        "--native-crops",
        action="store_true",
        help="Also retain three 512px native neutral/graded crop pairs before TIFF cleanup",
    )
    args = parser.parse_args(argv)
    report = benchmark(args)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report": str((Path(args.output) / "report.json").resolve()),
                "dimensions": report["native_dimensions"],
                "timings_seconds": report["timings_seconds"],
                "native_tiff_deleted": report["native_tiff_deleted"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
