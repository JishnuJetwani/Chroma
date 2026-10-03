"""Evaluation reports with per-image records and scene-clustered uncertainty."""

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .metrics import clipping_metrics, per_image_metrics, scene_bootstrap


def _sync(device) -> None:
    kind = torch.device(device).type
    if kind == "mps":
        torch.mps.synchronize()
    elif kind == "cuda":
        torch.cuda.synchronize(device)


def _scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _forward(model, image, style, valid_mask):
    return model(image, style)


def _identifier(batch, key, index, fallback):
    value = batch.get(key)
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    item = value[index]
    if isinstance(item, torch.Tensor):
        return str(item.item()) if item.numel() == 1 else str(item.tolist())
    return str(item)


def _image(tensor: torch.Tensor) -> Image.Image:
    pixels = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(pixels)


def _save_comparison(image, target, prediction, valid_mask, path: Path) -> None:
    panels = [_image(value) for value in (image, prediction, target)]
    if valid_mask is not None:
        valid = valid_mask[0].detach().cpu().numpy() > 0
        ys, xs = np.where(valid)
        if len(xs):
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            panels = [panel.crop(box) for panel in panels]
    width, height = panels[0].size
    sheet = Image.new("RGB", (width * 3, height + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (panel, label) in enumerate(zip(panels, ("Input", "Model", "Human target"), strict=False)):
        sheet.paste(panel, (index * width, 24))
        draw.text((index * width + 5, 5), label, fill="black")
    sheet.save(path)


def evaluate(
    model,
    loader,
    device,
    output_dir,
    checkpoint=None,
    protocol: dict | None = None,
    max_batches: int | None = None,
    save_images: bool = False,
    seed: int = 42,
    bootstrap_samples: int = 2000,
) -> dict:
    """Evaluate the loader as supplied; never silently change its split or resize policy.

    Reports include input and model agreement with paired expert targets.
    """
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if save_images:
        (destination / "comparisons").mkdir(exist_ok=True)
    rows = []
    total_seconds = 0.0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                image = batch["image"].to(device)
                target = batch["target"].to(device)
                style = batch.get("style_id", torch.zeros(len(image), dtype=torch.long)).to(device)
                valid_mask = batch.get("valid_mask")
                if valid_mask is not None:
                    valid_mask = valid_mask.to(device)
                _sync(device)
                started = time.perf_counter()
                prediction = _forward(model, image, style, valid_mask)
                _sync(device)
                elapsed = time.perf_counter() - started
                total_seconds += elapsed
                values = per_image_metrics(prediction, target, valid_mask)
                values.update(
                    {
                        "input_" + key: value
                        for key, value in per_image_metrics(image, target, valid_mask).items()
                    }
                )
                values.update(clipping_metrics(prediction, image, valid_mask))
                for index in range(len(image)):
                    number = len(rows)
                    row = {
                        "image_index": number,
                        "scene_id": _identifier(batch, "scene_id", index, f"image-{number:06d}"),
                        "style_id": int(style[index].item()),
                        "inference_seconds_per_image": elapsed / len(image),
                    }
                    row.update({key: _scalar(value[index].item()) for key, value in values.items()})
                    rows.append(row)
                    if save_images:
                        scene_digest = hashlib.sha256(row["scene_id"].encode()).hexdigest()[:8]
                        filename = f"{number:06d}_{scene_digest}.png"
                        _save_comparison(
                            image[index],
                            target[index],
                            prediction[index],
                            valid_mask[index] if valid_mask is not None else None,
                            destination / "comparisons" / filename,
                        )
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("Evaluation loader produced no images")
    metadata_keys = {"image_index", "scene_id", "style_id"}
    metric_names = sorted(set().union(*(row.keys() for row in rows)) - metadata_keys)
    aggregate = {}
    for name in metric_names:
        values = [row.get(name) if row.get(name) is not None else float("nan") for row in rows]
        aggregate[name] = scene_bootstrap(values, [row["scene_id"] for row in rows], seed, bootstrap_samples)
        finite = [v for v in values if math.isfinite(v)]
        aggregate[name]["image_mean"] = float(np.mean(finite)) if finite else None
    report = {
        "images": len(rows),
        "device": str(device),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": None,
        "model_config": asdict(model.config)
        if hasattr(model, "config") and is_dataclass(model.config)
        else None,
        "torch_version": str(torch.__version__),
        "protocol": protocol or {},
        "metrics": aggregate,
        "model_seconds": total_seconds,
        "timing_note": "Synchronized model-only batch timing; includes first-call warmup, excludes decoding, metrics, controls and encoding. Not full-resolution unless loader is full-resolution.",
        "metric_version": "srgb-metrics-v2-valid-ssim",
        "metric_note": "Encoded sRGB [0,1]; PSNR ceiling 120 dB; SSIM v2 Gaussian 11/1.5 (shrunk for small inputs), population covariance, reflect padding and fully valid windows (every support pixel mask>=1-1e-6); D65 Lab CIE76 (not CIEDE2000). Equal-weight scene bootstrap CIs. Empty masks/no fully valid SSIM windows omitted, not scored zero.",
    }
    if checkpoint and Path(checkpoint).is_file():
        digest = hashlib.sha256()
        with Path(checkpoint).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        report["checkpoint_sha256"] = digest.hexdigest()
    (destination / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fields = ["image_index", "scene_id", "style_id"] + metric_names
    with (destination / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Evaluation",
        "",
        f"Images: {len(rows)}. Device: `{device}`.",
        "",
        "Checkpoint: `%s`." % (checkpoint or "unspecified; do not call this a reproduced checkpoint"),
        "",
        "| Metric | Equal-weight scene mean | 95% scene bootstrap CI |",
        "|---|---:|---:|",
    ]
    for name, statistics in aggregate.items():
        if statistics["mean"] is not None:
            lines.append(
                "| {} | {:.6g} | [{:.6g}, {:.6g}] |".format(
                    name, statistics["mean"], statistics["ci_low"], statistics["ci_high"]
                )
            )
    lines.extend(
        [
            "",
            report["metric_note"],
            "",
            report["timing_note"],
            "",
            "Protocol:",
            "",
            "```json",
            json.dumps(report["protocol"], indent=2),
            "```",
            "",
        ]
    )
    (destination / "report.md").write_text("\n".join(lines))
    return report
