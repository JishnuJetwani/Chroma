"""Measure the actual cached-preview and original-resolution export path."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

from .inference import Grader
from .runtime import atomic_json, environment_info, file_sha256, synchronize


def benchmark_inference(checkpoint, image, output_dir, device="auto", repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Choose an empty benchmark output directory.")
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    source_hash = file_sha256(image)
    grader = Grader(checkpoint, device)
    analysis = grader.analyze(image)
    grader.render_preview(analysis)  # Warm up the cached rendering path.
    previews, exports = [], []
    for index in range(repeats):
        synchronize(grader.device)
        start = time.perf_counter()
        grader.render_preview(analysis)
        synchronize(grader.device)
        previews.append(time.perf_counter() - start)
        synchronize(grader.device)
        start = time.perf_counter()
        record = grader.export(analysis, destination / f"export-{index:02d}.png")
        synchronize(grader.device)
        record["end_to_end_seconds"] = time.perf_counter() - start
        exports.append(record)
    if file_sha256(image) != source_hash:
        raise RuntimeError("The source file changed during this benchmark.")
    result = {
        "environment": environment_info(grader.device),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source": str(Path(image).resolve()),
        "source_sha256": source_hash,
        "source_dimensions": list(analysis.original.size),
        "preview_dimensions": list(analysis.preview.size),
        "analysis_seconds": analysis.analysis_seconds,
        "cached_preview_seconds": previews,
        "cached_preview_median_seconds": statistics.median(previews),
        "exports": exports,
        "export_median_seconds": statistics.median(row["end_to_end_seconds"] for row in exports),
        "timing_note": "Local workload; one cold analysis, one preview warmup, repeated cached renders. "
        "Export includes full-resolution tensor conversion, tiled rendering and PNG encoding, "
        "but excludes initial image decoding and model loading. This is a performance fixture, "
        "not an image-quality evaluation or peak-memory measurement.",
    }
    if grader.device.type == "mps":
        result["mps_driver_bytes_at_end"] = torch.mps.driver_allocated_memory()
    atomic_json(destination / "benchmark.json", result)
    return result
