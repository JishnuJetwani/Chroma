"""Reproducible RAW pilot comparisons; no outer-test evaluation or JPEG weights."""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .metrics import per_image_metrics
from .raw_acquisition import INNER_VAL_COUNT, PARTITION_VERSION, SELECTION_VERSION, _rank
from .raw_data import RENDER_VERSION
from .raw_io import RAW_PIPELINE_VERSION
from .raw_model import RawGradingModel, RawModelConfig, baseline_raw, raw_supervised_loss, render_raw

TRUSTED_REGISTRY = Path(__file__).resolve().parents[2] / "configs/splits/cloud_full_7030.json"


def _verified_plan_records(plan):
    """Anchor copied acquisition labels to the separately mounted frozen split."""
    registry_bytes = TRUSTED_REGISTRY.read_bytes()
    if plan.get("registry_sha256") != hashlib.sha256(registry_bytes).hexdigest():
        raise ValueError("RAW plan registry hash differs from the trusted frozen registry")
    if (
        plan.get("partition_version") != PARTITION_VERSION
        or plan.get("selection_version") != SELECTION_VERSION
        or plan.get("outer_test_used") is not False
    ):
        raise ValueError("Unverified RAW pilot partition policy")
    registry = json.loads(registry_bytes)
    outer_train = registry["outer_splits"]["train"]
    outer_test = registry["outer_splits"]["test"]
    if (
        len(outer_train) != 1924
        or len(set(outer_train)) != 1924
        or len(outer_test) != 824
        or len(set(outer_test)) != 824
        or set(outer_train) & set(outer_test)
    ):
        raise ValueError("Trusted RAW outer scene registry is inconsistent")
    inner_val = set(sorted(outer_train, key=lambda sid: _rank(sid, PARTITION_VERSION))[:INNER_VAL_COUNT])
    pools = {"train": set(outer_train) - inner_val, "val": inner_val}
    selected = {}
    for split, pool in pools.items():
        count = plan.get("selected_counts", {}).get(split)
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= len(pool):
            raise ValueError("Invalid RAW plan selection count")
        for scene_id in sorted(pool, key=lambda sid: _rank(sid, SELECTION_VERSION))[:count]:
            selected[scene_id] = split
    records = plan.get("records", [])
    expected = {}
    for record in records:
        scene_id = record["scene_id"]
        if (
            scene_id in expected
            or scene_id not in selected
            or record.get("outer_split") != "train"
            or record.get("split") != selected[scene_id]
            or record.get("style_id") != 0
        ):
            raise ValueError("RAW plan differs from the frozen scene partition or nested selection")
        expected[scene_id] = record
    if set(expected) != set(selected):
        raise ValueError("RAW plan is missing selected scenes")
    return expected


def _training_generators(device, seed):
    """Corruption draws cannot alter the common scene sequence across arms."""
    return (
        torch.Generator(device=device).manual_seed(seed),
        torch.Generator(device=device).manual_seed(seed + 1),
    )


def _snapshot_sources(output):
    """Copy the executing pipeline source, including an uncommitted deployment."""
    destination = Path(output) / "source"
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("raw_io.py", "raw_data.py", "raw_model.py", "raw_training.py"):
        payload = Path(__file__).with_name(name).read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        (destination / name).write_bytes(payload)
    (destination / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return hashes


PACKED_CACHE_VERSION = 1


def _cache_context(cache):
    cache = Path(cache)
    manifest_bytes = (cache / "manifest.jsonl").read_bytes()
    plan_bytes = (cache / "plan.json").read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
    ids = [r["scene_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scene in RAW pilot")
    expected = _verified_plan_records(json.loads(plan_bytes))
    for row in rows:
        reference = expected.get(row["scene_id"])
        if (
            reference is None
            or reference["outer_split"] != "train"
            or reference["split"] != row["split"]
            or row["render_version"] != RENDER_VERSION
            or row.get("style_id") != 0
        ):
            raise ValueError("Unverified RAW scene split or rendering version")
        if (cache / row["path"]).resolve().parent != cache.resolve():
            raise ValueError("Cache path must be local to the manifest")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val")}
    if counts["train"] < 16 or counts["val"] < 8:
        raise ValueError("Need at least 16 training and 8 separate validation scenes")
    fingerprint = {
        "packed_cache_version": PACKED_CACHE_VERSION,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "trusted_registry_sha256": hashlib.sha256(TRUSTED_REGISTRY.read_bytes()).hexdigest(),
        "raw_pipeline_version": RAW_PIPELINE_VERSION,
        "render_version": RENDER_VERSION,
    }
    return rows, fingerprint


def _validate_sample(row, sample, metadata):
    keys = (
        "scene_id",
        "split",
        "style_id",
        "render_version",
        "training_size",
        "content_box",
        "raw",
        "target",
        "geometry",
    )
    if (
        not isinstance(metadata, dict)
        or any(key not in metadata or key not in row or metadata[key] != row[key] for key in keys)
        or metadata["training_size"] != 256
        or not isinstance(metadata["raw"], dict)
        or metadata["raw"].get("pipeline_version") != RAW_PIPELINE_VERSION
    ):
        raise ValueError("Embedded RAW cache metadata differs from the manifest or current pipeline")
    if set(sample) != {"image", "target", "valid_mask"} or any(
        not isinstance(value, torch.Tensor) for value in sample.values()
    ):
        raise ValueError("Invalid float RAW pair tensor schema")
    x, y, m = (sample[k] for k in ("image", "target", "valid_mask"))
    if (
        any(value.dtype != torch.float32 for value in sample.values())
        or x.shape != y.shape
        or x.shape != (3, 256, 256)
        or m.shape != (1, 256, 256)
        or not all(torch.isfinite(v).all() for v in sample.values())
        or y.min() < -1e-6
        or y.max() > 1 + 1e-6
        or m.min() < 0
        or m.max() > 1
        or m.sum() == 0
    ):
        details = {
            k: {
                "dtype": str(v.dtype),
                "shape": list(v.shape),
                "min": float(v.min()) if v.numel() else None,
                "max": float(v.max()) if v.numel() else None,
            }
            for k, v in sample.items()
        }
        raise ValueError(f"Invalid float RAW pair {row['scene_id']}: {details}")
    # Preserve floating scene-linear RAW negatives/headroom. Only display-target
    # antialiasing roundoff within one micro-unit is normalized.
    sample["target"] = y.clamp(0, 1)
    box = metadata["content_box"]
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(not isinstance(value, int) or isinstance(value, bool) for value in box)
    ):
        raise ValueError("Invalid RAW cache content rectangle")
    left, top, right, bottom = box
    if not 0 <= left < right <= 256 or not 0 <= top < bottom <= 256:
        raise ValueError("Invalid RAW cache content rectangle")
    expected_mask = torch.zeros_like(m)
    expected_mask[:, top:bottom, left:right] = 1
    if not torch.equal(m, expected_mask):
        raise ValueError("RAW cache validity mask differs from its content rectangle")
    return sample


def _read_npz_sample(cache, row):
    # A single volume read supplies both checksum and decoding, avoiding a
    # second remote filesystem access for every compressed member.
    payload = (Path(cache) / row["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError("RAW cache checksum mismatch: " + row["scene_id"])
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        sample = {key: torch.from_numpy(data[key]) for key in ("image", "target", "valid_mask")}
        metadata = json.loads(str(data["metadata"]))
    return row, _validate_sample(row, sample, metadata)


def _pack_items(items):
    result = {}
    for split in ("train", "val"):
        values = [(row, sample) for row, sample in items if row["split"] == split]
        result[split] = {
            key: torch.stack([sample[key] for _, sample in values])
            for key in ("image", "target", "valid_mask")
        }
        result[split]["rows"] = [row for row, _ in values]
    return result


def _load_npz_cache(cache, rows, workers=8):
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 8:
        raise ValueError("RAW cache verification supports one to eight workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # executor.map preserves manifest order for reproducible training draws.
        items = list(pool.map(lambda row: _read_npz_sample(cache, row), rows))
    return _pack_items(items)


def prepare_packed_cache(cache, *, workers=8):
    """Validate source NPZs on CPU once and persist an atomic tensor snapshot.

    The receipt binds the packed file to the manifest, acquisition plan, trusted
    split registry and pipeline versions. It certifies the validated snapshot;
    fast loads use this snapshot, not potentially changed unused NPZ files.
    """
    cache = Path(cache)
    started = time.monotonic()
    rows, fingerprint = _cache_context(cache)
    packed = _load_npz_cache(cache, rows, workers)
    path, temporary = cache / "packed.pt", cache / "packed.pt.partial"
    try:
        torch.save({"fingerprint": fingerprint, "splits": packed}, temporary)
        receipt = {
            **fingerprint,
            "packed_sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
            "packed_bytes": temporary.stat().st_size,
            "counts": {split: len(data["rows"]) for split, data in packed.items()},
            "validation": "All source NPZ checksums, embedded metadata, float arrays and masks checked",
            "workers": workers,
            "elapsed_seconds": time.monotonic() - started,
        }
        temporary.replace(path)
        receipt_temporary = cache / "packed.source.json.partial"
        receipt_temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt_temporary.replace(cache / "packed.source.json")
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def _load_packed_cache(cache, rows, fingerprint):
    path, receipt_path = cache / "packed.pt", cache / "packed.source.json"
    if not path.is_file() or not receipt_path.is_file():
        raise ValueError("Incomplete packed RAW cache; prepare it again on CPU")
    if path.resolve().parent != cache.resolve() or receipt_path.resolve().parent != cache.resolve():
        raise ValueError("Packed RAW cache must be local to the manifest")
    receipt = json.loads(receipt_path.read_text())
    if any(receipt.get(key) != value for key, value in fingerprint.items()):
        raise ValueError("Packed RAW receipt differs from current manifest, plan, registry or pipeline")
    payload = path.read_bytes()
    if len(payload) != receipt.get("packed_bytes") or hashlib.sha256(payload).hexdigest() != receipt.get(
        "packed_sha256"
    ):
        raise ValueError("Packed RAW cache checksum mismatch")
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("fingerprint") != fingerprint:
        raise ValueError("Packed RAW embedded fingerprint mismatch")
    packed = saved.get("splits")
    if not isinstance(packed, dict) or set(packed) != {"train", "val"}:
        raise ValueError("Packed RAW split schema mismatch")
    for split, data in packed.items():
        expected_rows = [row for row in rows if row["split"] == split]
        if not isinstance(data, dict) or data.get("rows") != expected_rows:
            raise ValueError("Packed RAW rows differ from the validated manifest")
        if set(data) != {"image", "target", "valid_mask", "rows"}:
            raise ValueError("Packed RAW tensor schema mismatch")
        if receipt.get("counts", {}).get(split) != len(expected_rows):
            raise ValueError("Packed RAW receipt count mismatch")
        for key in ("image", "target", "valid_mask"):
            value = data[key]
            channels = 1 if key == "valid_mask" else 3
            if not isinstance(value, torch.Tensor) or value.shape != (len(expected_rows), channels, 256, 256):
                raise ValueError("Packed RAW tensor shape mismatch")
        for index, row in enumerate(expected_rows):
            sample = {key: data[key][index] for key in ("image", "target", "valid_mask")}
            _validate_sample(row, sample, row)
            data["target"][index].copy_(sample["target"])
    return packed


def load_cache(cache, device):
    cache = Path(cache)
    rows, fingerprint = _cache_context(cache)
    if (cache / "packed.pt").exists() or (cache / "packed.source.json").exists():
        packed = _load_packed_cache(cache, rows, fingerprint)
    else:
        if torch.device(device).type == "cuda":
            raise ValueError("Prepare packed RAW cache on CPU before allocating GPU training")
        packed = _load_npz_cache(cache, rows)
    for data in packed.values():
        for key in ("image", "target", "valid_mask"):
            data[key] = data[key].to(device)
    return packed, fingerprint["manifest_sha256"]


def local_fault(image, generator, *, family="ellipse", valid_mask=None):
    """Known unclipped scene-linear exposure/veil faults with compact support."""
    b, _, h, w = image.shape

    def rand(*shape):
        return torch.rand(shape, device=image.device, generator=generator)

    yy = torch.linspace(-1, 1, h, device=image.device)[None, None, :, None]
    xx = torch.linspace(-1, 1, w, device=image.device)[None, None, None, :]
    cx, cy = (rand(b, 1, 1, 1) - 0.5) * 1.4, (rand(b, 1, 1, 1) - 0.5) * 1.4
    rx, ry = 0.25 + 0.5 * rand(b, 1, 1, 1), 0.25 + 0.5 * rand(b, 1, 1, 1)
    if family == "ellipse":
        distance = ((xx - cx) / rx).square() + ((yy - cy) / ry).square()
        support = (1 - distance).clamp(0, 1).square()
    elif family == "stripe":
        # A held-out family: feathered diagonal stripe, never used for training.
        distance = ((xx + 0.7 * yy - cx) / rx).abs()
        support = (1 - distance).clamp(0, 1).square()
    else:
        raise ValueError("Unknown corruption family")
    if valid_mask is not None:
        if valid_mask.shape != image[:, :1].shape:
            raise ValueError("Corruption validity must be B1HW")
        support = support * valid_mask
    ev = (rand(b, 1, 1, 1) - 0.5) * 1.5
    veil = 0.08 * rand(b, 3, 1, 1)
    corrupted = image * torch.exp2(support * ev) + support * veil
    return corrupted, support


def masked_l1(a, b, mask):
    return ((a - b).abs() * mask).sum() / (3 * mask.sum()).clamp_min(1)


@torch.inference_mode()
def evaluate(model, data, output=None):
    records, fault_records = [], []
    generator = torch.Generator(device=data["image"].device).manual_seed(7821)
    for start in range(0, len(data["rows"]), 8):
        end = start + 8
        x, y, mask = (data[k][start:end] for k in ("image", "target", "valid_mask"))
        prediction = baseline_raw(x) if model is None else model(x, valid_mask=mask)
        metrics = per_image_metrics(prediction, y, mask)
        corrupt, support = local_fault(x, generator, family="stripe", valid_mask=mask)
        edited = baseline_raw(corrupt) if model is None else model(corrupt, valid_mask=mask)
        for i, row in enumerate(data["rows"][start:end]):
            records.append({"scene_id": row["scene_id"], **{k: float(v[i]) for k, v in metrics.items()}})
            inside, outside = (
                mask[i : i + 1] * (support[i : i + 1] > 0.05),
                mask[i : i + 1] * (support[i : i + 1] == 0),
            )
            fault_records.append(
                {
                    "scene_id": row["scene_id"],
                    "corrupted_target_mae_inside": float(masked_l1(edited[i : i + 1], y[i : i + 1], inside)),
                    "clean_target_mae_inside": float(masked_l1(prediction[i : i + 1], y[i : i + 1], inside)),
                    "outside_output_drift": float(
                        masked_l1(edited[i : i + 1], prediction[i : i + 1], outside)
                    ),
                }
            )
            if output is not None and start < 8:
                left, top, right, bottom = row["content_box"]
                images = [baseline_raw(x[i : i + 1])[0], prediction[i], y[i]]
                strip = torch.cat([p[:, top:bottom, left:right] for p in images], -1)
                pixels = strip.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).save(
                    Path(output) / (row["scene_id"].replace(":", "-") + ".png")
                )
    mean = {k: sum(row[k] for row in records) / len(records) for k in records[0] if k != "scene_id"}
    faults = {
        k: sum(row[k] for row in fault_records) / len(fault_records)
        for k in fault_records[0]
        if k != "scene_id"
    }
    return {
        "count": len(records),
        "mean": mean,
        "per_scene": records,
        "held_out_stripe_faults": {"mean": faults, "per_scene": fault_records},
    }


def train_pilot(cache, output, *, steps=800, budget_seconds=900, device="cuda", commit=lambda: None):
    """Matched-update arms, fixed seed; validation findings are exploratory."""
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("Run already completed; use a new run ID")
    source_hashes = _snapshot_sources(output)
    plan_hash = hashlib.sha256((Path(cache) / "plan.json").read_bytes()).hexdigest()
    registry_hash = hashlib.sha256(TRUSTED_REGISTRY.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    data, manifest_hash = load_cache(cache, device)
    preload_seconds = time.monotonic() - started
    baseline = evaluate(None, data["val"])
    configs = {
        "global": RawModelConfig(max_local_exposure_ev=0, max_local_log_contrast=0),
        "shared": RawModelConfig(),
        "preservation": RawModelConfig(),
        "robust": RawModelConfig(enable_veil=True),
    }
    report = {
        "format_version": 1,
        "protocol": "RAW pilot inner validation; all scenes from frozen outer train",
        "seed": 20260919,
        "requested_steps_per_arm": steps,
        "budget_seconds": budget_seconds,
        "manifest_sha256": manifest_hash,
        "raw_pipeline_version": RAW_PIPELINE_VERSION,
        "source_sha256": source_hashes,
        "source_snapshot": "source/",
        "plan_sha256": plan_hash,
        "trusted_registry_sha256": registry_hash,
        "render_version": RENDER_VERSION,
        "preload_seconds": preload_seconds,
        "counts": {s: len(d["rows"]) for s, d in data.items()},
        "baseline": baseline,
        "arms": {},
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 0.0002,
            "weight_decay": 0.0001,
            "batch_size": 16,
            "precision": "FP32 renderer; BF16 CNN on CUDA",
            "sampling": "Same scene-sampling seed in each arm; independent corruption RNG seed+1",
            "augmentation": "preservation and robust: every fourth batch uses unclipped local exposure/veil faults + .1 outside-output consistency; robust additionally has a veil head",
        },
        "device": torch.cuda.get_device_name() if device.type == "cuda" else str(device),
    }
    (output / "progress.json").write_text(json.dumps(report, indent=2))
    commit()
    for name, config in configs.items():
        if time.monotonic() - started > budget_seconds - 30:
            break
        torch.manual_seed(report["seed"])
        sampler, fault_generator = _training_generators(device, report["seed"])
        model = RawGradingModel(config).to(device)
        if device.type == "cuda":
            model.set_cnn_autocast("bf16")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0001)
        history = []
        arm_started = time.monotonic()
        completed = 0
        model.train()
        train = data["train"]
        folder = output / name
        folder.mkdir(exist_ok=True)
        for step in range(steps):
            if time.monotonic() - started > budget_seconds - 20:
                break
            index = torch.randint(len(train["rows"]), (16,), device=device, generator=sampler)
            x, y, mask = (train[k][index] for k in ("image", "target", "valid_mask"))
            preservation = x.new_zeros(())
            if name in ("preservation", "robust") and step % 4 == 3:
                with torch.no_grad():
                    clean = model(x, valid_mask=mask)
                corrupted, support = local_fault(x, fault_generator, valid_mask=mask)
                recipe = model.analyze(corrupted, valid_mask=mask)
                prediction = render_raw(corrupted, recipe)
                preservation = masked_l1(prediction, clean, mask * (support == 0))
            else:
                recipe = model.analyze(x, valid_mask=mask)
                prediction = render_raw(x, recipe)
            terms = raw_supervised_loss(prediction, y, recipe, valid_mask=mask)
            loss = terms["total"] + 0.1 * preservation
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite RAW loss at {name}:{step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            completed = step + 1
            # A fixed cosine schedule, never selected by the validation score.
            lr = 0.0002 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * completed / steps)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            if completed % 100 == 0 or completed == 1:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                event = {
                    "arm": name,
                    "step": completed,
                    "loss": float(loss.detach()),
                    "rgb_l1": float(terms["rgb_l1"].detach()),
                    "outside_consistency": float(preservation.detach()),
                    "gradient_norm": float(norm),
                    "elapsed_seconds": time.monotonic() - arm_started,
                }
                history.append(event)
                print(json.dumps(event), flush=True)
                (output / "live.json").write_text(json.dumps(event))
                commit()
        model.eval()
        # Validate the FP32 inference path used by the local app, independent of
        # the optional reduced-precision convolution training optimization.
        model.set_cnn_autocast("none")
        checkpoint = {
            "model": model.state_dict(),
            "model_config": asdict(config),
            "raw_pipeline_version": RAW_PIPELINE_VERSION,
            "metadata": {
                "trained": True,
                "stage": "RAW pilot",
                "arm": name,
                "steps": completed,
                "manifest_sha256": manifest_hash,
                "train_scenes": report["counts"]["train"],
                "source_sha256": source_hashes,
                "plan_sha256": plan_hash,
                "trusted_registry_sha256": registry_hash,
                "validation_scenes": report["counts"]["val"],
                "seed": report["seed"],
            },
        }
        torch.save(checkpoint, folder / "model.pt")
        validation = evaluate(model, data["val"], folder)
        result = {
            "steps": completed,
            "converged": False,
            "parameters": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": time.monotonic() - arm_started,
            "history": history,
            "validation": validation,
            "checkpoint_sha256": hashlib.sha256((folder / "model.pt").read_bytes()).hexdigest(),
        }
        report["arms"][name] = result
        (folder / "result.json").write_text(json.dumps(result, indent=2))
        (output / "progress.json").write_text(json.dumps(report, indent=2))
        commit()
        del optimizer, model
    report["elapsed_seconds"] = time.monotonic() - started
    report["complete"] = len(report["arms"]) == len(configs) and all(
        a["steps"] == steps for a in report["arms"].values()
    )
    report["selection_note"] = (
        "No claim of benchmark superiority; inspect real RAW previews before promotion."
    )
    if device.type == "cuda":
        report["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    (output / "report.json").write_text(json.dumps(report, indent=2))
    shutil.copy2(Path(cache) / "preparation.json", output / "preparation.json")
    commit()
    return {k: v for k, v in report.items() if k not in ("baseline", "arms")} | {
        "baseline": baseline["mean"],
        "arms": {
            k: {
                "steps": v["steps"],
                "seconds": v["elapsed_seconds"],
                "validation": v["validation"]["mean"],
                "faults": v["validation"]["held_out_stripe_faults"]["mean"],
            }
            for k, v in report["arms"].items()
        },
    }
