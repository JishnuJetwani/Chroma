"""A bounded, fresh 70/30 training curriculum on a single CUDA device.

Performance selection uses training images only. Model selection never uses
the held-out 30%; evaluation runs after the full fixed schedule completes.
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import torch

from .runtime import atomic_json, file_sha256, load_checkpoint, save_checkpoint, seed_everything

SEED = 20260918
MANIFEST_SHA256 = "ddb01f569477fae9be510830fb7d154bb7d99ac2bc7c97393215a0106f191a3b"
STAGES = (("global", 10), ("repair", 5), ("joint", 10))
# Account billing API rates read on 2026-10-12; estimate, not a billing receipt.
HOURLY_USD = 3.95 + 4 * 0.04730 + 16 * 0.00800


class RuntimeBudgetExceeded(RuntimeError):
    pass


def _check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise RuntimeBudgetExceeded("GPU runtime safety budget reached")


def stage_config(cache, output, stage, epochs, batch_size, precision, max_seconds):
    return {
        "model": {
            "architecture": "global" if stage == "global" else "local",
            "styles": ["FiveK Expert C", "FiveK Expert A", "FiveK Expert B"],
            "preview_size": 256,
            "lut_size": 33,
            "num_luts": 8,
            "map_size": 64,
            "upsampling": "guided",
            "order": "global_first",
        },
        "data": {"manifest": str(Path(cache) / "manifest.jsonl"), "image_size": 256, "limit": None},
        "training": {
            "stage": stage,
            "epochs": epochs,
            "batch_size": batch_size,
            "accumulation_steps": 8 // batch_size,
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "seed": SEED,
            "device": "cuda",
            "validation": False,
            "num_workers": 0,
            "cpu_threads": 4,
            "preservation": True,
            "encoder_autocast": precision,
            "max_steps": None,
            "max_seconds": max_seconds,
            "log_every": 100,
        },
        "loss": {},
        "output_dir": str(output),
    }


def benchmark_candidate(dataset, cache, output, batch_size, precision, deadline=None):
    from .models import ModelConfig, build_model
    from .training import _loss, _resolve

    result = {
        "batch_size": batch_size,
        "accumulation_steps": 8 // batch_size,
        "encoder_autocast": precision,
        "stages": {},
        "valid": True,
    }
    for stage, epochs in STAGES:
        seed_everything(SEED)
        config = _resolve(stage_config(cache, output, stage, epochs, batch_size, precision, None))
        model = build_model(ModelConfig(**config["model"])).cuda().train()
        model.set_cnn_autocast(precision)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        iterator = dataset.iter_batches(batch_size, True, SEED, "cuda")
        warmup, measured = 8, 32
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = None
        for index in range(warmup + measured):
            _check_deadline(deadline)
            if index == warmup:
                torch.cuda.synchronize()
                started = time.perf_counter()
            batch = next(iterator)
            loss, _ = _loss(model, batch, config["training"], config["loss"], index)
            torch._assert_async(torch.isfinite(loss), "Nonfinite benchmark loss")
            (loss / config["training"]["accumulation_steps"]).backward()
            if (index + 1) % config["training"]["accumulation_steps"] == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch._assert_async(torch.isfinite(norm), "Nonfinite benchmark gradient")
                optimizer.step()
                if hasattr(model, "constrain_parameters"):
                    model.constrain_parameters()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        # Compare both precision modes using the SAME nontrivially updated model.
        model.eval()
        with torch.no_grad():
            model.set_cnn_autocast("none")
            reference = model(batch["image"], batch["style_id"])
            model.set_cnn_autocast(precision)
            candidate = model(batch["image"], batch["style_id"])
            difference = (reference - candidate).abs()
            mean_error, max_error = difference.mean().item(), difference.max().item()
        valid = math.isfinite(mean_error) and mean_error <= 0.001 and max_error <= 0.02
        result["valid"] &= valid
        row = {
            "measured_microbatches": measured,
            "warmup_microbatches": warmup,
            "seconds": seconds,
            "images_per_second": measured * batch_size / seconds,
            "seconds_per_microbatch": seconds / measured,
            "estimated_stage_seconds": math.ceil(len(dataset) / batch_size) * epochs * seconds / measured,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "precision_output_mae": mean_error,
            "precision_output_max_error": max_error,
            "precision_check_passed": valid,
        }
        result["stages"][stage] = row
        print(
            json.dumps(
                {"benchmark": {"batch_size": batch_size, "precision": precision, "stage": stage, **row}}
            ),
            flush=True,
        )
        del model, optimizer, loss, batch, reference, candidate, difference
        gc.collect()
        torch.cuda.empty_cache()
    result["estimated_training_seconds"] = sum(
        row["estimated_stage_seconds"] for row in result["stages"].values()
    )
    return result


def run(cache: Path, output: Path, budget_seconds=3600, commit=lambda: None, resume=False):
    from .cloud_data import CachedPairedDataset
    from .training import train

    if not 60 <= budget_seconds <= 3600:
        raise ValueError("Initial GPU budget must be 60..3600 seconds")
    if output.exists() and not resume:
        raise FileExistsError("Fresh split training requires a new run directory")
    if resume and not (output / "run.json").is_file():
        raise FileNotFoundError("No saved cloud run to resume")
    if not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA")
    started = time.perf_counter()
    previous_report = json.loads((output / "run.json").read_text()) if resume else None
    previous_seconds = previous_report["elapsed_seconds"] if resume else 0
    deadline = started + budget_seconds - previous_seconds
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    # Preserve FP32 matmul semantics; BF16 is restricted to the CNN if it wins.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=resume)
    index = json.loads((cache / "index.json").read_text())
    if (
        index["manifest_sha256"] != MANIFEST_SHA256
        or file_sha256(cache / "manifest.jsonl") != MANIFEST_SHA256
    ):
        raise ValueError("Manifest differs from the frozen, locally verified public 70/30 split")
    if index["seed"] != SEED or not index["requires_training_from_scratch"]:
        raise ValueError("Unexpected split seed or initialization policy")
    if index["outer_scene_counts"] != {"train": 1924, "test": 824} or index["inner_val_fraction"] != 0:
        raise ValueError("Unexpected split: expected fixed 1924/824 scenes, no validation")
    if index["row_counts"] != {"train": 1958, "val": 0, "test": 838}:
        raise ValueError("Unexpected number of unique scene/style pairs")
    report = {
        "status": "preparing",
        "seed": SEED,
        "schedule": dict(STAGES),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "manifest_sha256": index["manifest_sha256"],
        "cache_index_sha256": file_sha256(cache / "index.json"),
        "scene_counts": index["outer_scene_counts"],
        "pair_counts": index["row_counts"],
        "selection": "Fixed final schedule; test is never used to select settings or checkpoints",
        "initialization": "From scratch; no prior pilot checkpoints",
        "budget_seconds": budget_seconds,
        "estimated_hourly_usd": HOURLY_USD,
        "cost_note": "Runtime estimate from account rates; excludes CPU preparation, startup/idle, storage, credits and billing adjustments",
        "source_sha256": {path.name: file_sha256(path) for path in Path(__file__).parent.glob("*.py")},
    }
    if resume:
        if previous_report["status"] == "completed":
            raise ValueError("Run already completed; do not retrain or reevaluate it")
        if (
            previous_report["manifest_sha256"] != index["manifest_sha256"]
            or previous_report["cache_index_sha256"] != report["cache_index_sha256"]
        ):
            raise ValueError("Cannot resume with changed data")
        if previous_report["seed"] != SEED or previous_report["schedule"] != dict(STAGES):
            raise ValueError("Cannot resume with a changed schedule")
        # Orchestration may be repaired, but learning and evaluation algorithms
        # must remain exactly those used by the earlier completed phases.
        for name in (
            "training.py",
            "runtime.py",
            "models.py",
            "renderer.py",
            "color.py",
            "corruptions.py",
            "data.py",
            "cloud_data.py",
            "metrics.py",
            "evaluation.py",
            "recipes.py",
        ):
            if previous_report["source_sha256"][name] != report["source_sha256"][name]:
                raise ValueError(f"Cannot resume after changing {name}")
        orchestration_sha = report["source_sha256"]["cloud_runner.py"]
        report = previous_report
        report.setdefault("resumptions", []).append(
            {"previous_recorded_seconds": previous_seconds, "cloud_runner_sha256": orchestration_sha}
        )

    def persist():
        report["elapsed_seconds"] = previous_seconds + time.perf_counter() - started
        report["estimated_compute_usd"] = report["elapsed_seconds"] * HOURLY_USD / 3600
        atomic_json(output / "run.json", report)
        commit()

    try:
        data = CachedPairedDataset(cache / "index.json", "train").preload_to("cuda")
        torch.cuda.synchronize()
        cache_seconds = time.perf_counter() - started
        report["cache_load_seconds"] = report.get("cache_load_seconds", 0) + cache_seconds
        print(
            json.dumps(
                {
                    "training_cache_loaded": len(data),
                    "gpu": report["gpu"],
                    "seconds": cache_seconds,
                    "resuming": resume,
                }
            ),
            flush=True,
        )
        if resume:
            selected = report["selected"]
        else:
            candidates = []
            for batch_size, precision in ((2, "none"), (8, "none"), (8, "bf16")):
                candidates.append(benchmark_candidate(data, cache, output, batch_size, precision, deadline))
            selected = min(
                (item for item in candidates if item["valid"]),
                key=lambda item: item["estimated_training_seconds"],
            )
            report.update(benchmark=candidates, selected=selected, status="benchmarked")
            atomic_json(output / "benchmark.json", {"candidates": candidates, "selected": selected})
        persist()
        # Leave five minutes for checkpoint overhead/final evaluation. A poor
        # measured runtime returns with evidence rather than starting a huge run.
        remaining = deadline - time.perf_counter()
        if not resume and selected["estimated_training_seconds"] * 1.15 > remaining - 300:
            report["status"] = "runtime_budget_gate"
            persist()
            return report
        report.setdefault("stages", {})
        initialization = None

        def checkpoint_callback(directory, record):
            print(json.dumps({"durable_checkpoint": str(directory), **record}), flush=True)
            commit()

        for stage, epochs in STAGES:
            stage_output = output / stage
            summary_path = stage_output / "summary.json"
            if resume and summary_path.is_file():
                saved = json.loads(summary_path.read_text())
                if (
                    saved["status"] == "completed"
                    and saved.get("final_selected")
                    and (stage_output / "final.pt").is_file()
                ):
                    if (
                        saved["completed_epoch"] != epochs
                        or saved["manifest_sha256"] != index["manifest_sha256"]
                    ):
                        raise ValueError("Saved completed phase differs from requested schedule/data")
                    report["stages"][stage] = saved
                    initialization = str(stage_output / "final.pt")
                    print(json.dumps({"reusing_completed_stage": stage, "epochs": epochs}), flush=True)
                    continue
            remaining = deadline - time.perf_counter() - 180
            if remaining <= 0:
                report["status"] = "budget_stopped"
                persist()
                return report
            config = stage_config(
                cache,
                output / stage,
                stage,
                epochs,
                selected["batch_size"],
                selected["encoder_autocast"],
                remaining,
            )
            print(
                json.dumps(
                    {
                        "starting_stage": stage,
                        "epochs": epochs,
                        "batch_size": selected["batch_size"],
                        "precision": selected["encoder_autocast"],
                    }
                ),
                flush=True,
            )
            resume_checkpoint = (
                stage_output / "last.pt" if resume and (stage_output / "last.pt").exists() else None
            )
            summary = train(
                config,
                resume=resume_checkpoint,
                initialize=None if resume_checkpoint else initialization,
                train_dataset=data,
                checkpoint_callback=checkpoint_callback,
            )
            report["stages"][stage] = summary
            if summary["status"] != "completed" or not summary.get("final_checkpoint"):
                report["status"] = "budget_stopped"
                persist()
                return report
            initialization = summary["final_checkpoint"]
            report["status"] = "training"
            persist()
        # Release the training cache before first access to test tensors.
        del data
        gc.collect()
        torch.cuda.empty_cache()
        report["training_complete_seconds"] = previous_seconds + time.perf_counter() - started
        report["status"] = "evaluating"
        persist()
        from .evaluation import evaluate

        test = CachedPairedDataset(cache / "index.json", "test").preload_to("cuda")
        report["evaluations"] = {}

        def test_batches():
            for batch in test.iter_batches(8, False, SEED, "cuda"):
                _check_deadline(deadline)
                yield batch

        for stage in ("global", "joint"):
            _check_deadline(deadline)
            checkpoint = output / stage / "final.pt"
            model, payload = load_checkpoint(checkpoint, "cuda")
            # Report normal FP32 inference, as used by the desktop app.
            metrics = evaluate(
                model,
                test_batches(),
                "cuda",
                output / "evaluation" / stage,
                checkpoint=checkpoint,
                protocol={
                    "split": "test",
                    "image_size": 256,
                    "batch_size": 8,
                    "manifest_sha256": index["manifest_sha256"],
                    "scene_count": 824,
                    "checkpoint_selection": "fixed_schedule_final",
                    "test_used_for_tuning": False,
                    "inference_precision": "float32",
                },
                include_controls=False,
                seed=SEED,
            )
            report["evaluations"][stage] = {key: value["mean"] for key, value in metrics["metrics"].items()}
            # Small portable inference artifact; full optimizer checkpoints stay
            # on the Volume for resumption instead of duplicating them locally.
            portable = output / stage / "model.pt"
            save_checkpoint(
                portable,
                model,
                epoch=payload["epoch"],
                step=payload["step"],
                training_config=payload["training_config"],
                metadata={**payload["metadata"], "source_checkpoint_sha256": file_sha256(checkpoint)},
            )
            report.setdefault("portable_models", {})[stage] = {
                "path": str(portable),
                "sha256": file_sha256(portable),
            }
            del model
            gc.collect()
        report["status"] = "completed"
        persist()
        return report
    except RuntimeBudgetExceeded:
        report["status"] = "budget_stopped"
        persist()
        return report
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        persist()
        raise
