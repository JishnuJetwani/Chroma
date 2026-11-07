"""Compare old/new models on identical full-FiveK test photos and fixed cohorts."""

import argparse
import csv
import json
import math
from pathlib import Path

from photograder.metrics import scene_bootstrap


def summarize(run_directory, registry_path):
    root = Path(run_directory)
    run = json.loads((root / "run.json").read_text())
    registry = json.loads(Path(registry_path).read_text())
    if run["status"] != "completed":
        raise ValueError("Training and all evaluations must complete before reporting")
    if registry["original_manifest_sha256"] != run["manifest_sha256"]:
        raise ValueError("Split registry does not match the evaluated dataset")
    heldout = set(registry["outer_splits"]["test"])
    if run["scene_counts"] != {"train": 3500, "test": 1500} or len(heldout) != 1500:
        raise ValueError("Expected the complete 3,500/1,500 experiment")
    records = {}
    for model in ("previous", "global", "joint"):
        with (root / "evaluation" / model / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        records[model] = {(row["scene_id"], int(row["style_id"])): row for row in rows}
        if len(rows) != len(records[model]):
            raise ValueError("Duplicate observations in evaluation")
        if {scene for scene, style in records[model] if style == 0} != heldout:
            raise ValueError("Evaluation does not cover every held-out Expert C scene")
    if not records["previous"].keys() == records["global"].keys() == records["joint"].keys():
        raise ValueError("Models were evaluated on different scene/style pairs")
    cohorts = {"all": heldout}
    cohorts.update({name: heldout & set(ids) for name, ids in registry["cohorts"].items()})
    report = {
        "manifest_sha256": run["manifest_sha256"],
        "primary_style": "FiveK Expert C",
        "scene_counts": run["scene_counts"],
        "pair_counts": run["pair_counts"],
        "schedule": run["schedule"],
        "gpu": run["gpu"],
        "cohorts": {},
        "comparison_checkpoint_sha256": run["comparison_checkpoint_sha256"],
        "elapsed_seconds": run["elapsed_seconds"],
        "estimated_training_function_usd": run["estimated_compute_usd"],
        "limitations": [
            "Same epochs but more optimizer updates than the subset run",
            "Custom 70/30 split and padded 256px metrics, not a published protocol",
            "Original holdout has been inspected before; additional cohort was newly reserved",
            "Single training seed; uncertainty is across test scenes, not training runs",
        ],
    }
    for cohort, scene_set in cohorts.items():
        scenes = sorted(scene_set)
        keys = [(scene, 0) for scene in scenes]
        group = {"scenes": len(scenes), "models": {}, "joint_minus_previous": {}}
        for label, source, prefix in (
            ("input", "joint", "input_"),
            ("previous", "previous", ""),
            ("global", "global", ""),
            ("joint", "joint", ""),
        ):
            group["models"][label] = {}
            for metric in ("psnr", "ssim", "delta_e76"):
                values = [float(records[source][key][prefix + metric]) for key in keys]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Missing or nonfinite primary metric")
                group["models"][label][metric] = scene_bootstrap(values, scenes, seed=run["seed"])
        for metric in ("psnr", "ssim", "delta_e76", "new_clipped_high_fraction"):
            differences = [
                float(records["joint"][key][metric]) - float(records["previous"][key][metric]) for key in keys
            ]
            if not all(math.isfinite(value) for value in differences):
                raise ValueError("Missing or nonfinite paired difference")
            group["joint_minus_previous"][metric] = scene_bootstrap(differences, scenes, seed=run["seed"])
        group["psnr_wins"] = sum(
            float(records["joint"][key]["psnr"]) > float(records["previous"][key]["psnr"]) for key in keys
        )
        report["cohorts"][cohort] = group
    (root / "full_fivek_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Complete FiveK JPEG experiment",
        "",
        "3,500 training / 1,500 held-out scenes. Existing assignments were preserved; only previously missing scenes were partitioned.",
        "",
        "Training uses all 3,500 Expert C pairs plus the unchanged 17 A and 17 B examples. Fixed curriculum: 10 global, 5 repair, 10 joint epochs; batch eight, FP32, seed 20260918. Final checkpoints are selected by schedule, with no test-based tuning.",
        "",
        "## Same-image Expert C comparison",
        "",
        "| Output | PSNR ↑ | SSIM ↑ | CIE76 color error ↓ |",
        "|---|---:|---:|---:|",
    ]
    for label in ("input", "previous", "global", "joint"):
        values = report["cohorts"]["all"]["models"][label]
        lines.append(
            f"| {label.title()} | {values['psnr']['mean']:.3f} dB | {values['ssim']['mean']:.4f} | {values['delta_e76']['mean']:.3f} |"
        )
    lines += [
        "",
        "Previous is the existing 2,748-scene experiment's final local model; global and joint are the newly trained full-data models. All rows above score the identical 1,500 held-out photos.",
        "",
        "## Original and additional holdouts",
        "",
        "| Cohort | Photos | Previous PSNR | New joint PSNR | Paired improvement (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort in ("all", "original", "additional"):
        group = report["cohorts"][cohort]
        delta = group["joint_minus_previous"]["psnr"]
        lines.append(
            f"| {cohort.title()} | {group['scenes']} | {group['models']['previous']['psnr']['mean']:.3f} | {group['models']['joint']['psnr']['mean']:.3f} | {delta['mean']:+.3f} [{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}] dB |"
        )
    lines += [
        "",
        "## Runtime and interpretation",
        "",
        f"Recorded training-function time including cache loading and all three evaluations: {run['elapsed_seconds'] / 60:.2f} minutes on {run['gpu']}. Estimated function compute: ${run['estimated_compute_usd']:.3f}; this is not a bill and excludes CPU preparation, image building, startup/idle, storage, and credits.",
        "",
    ]
    lines.extend(f"- {item}." for item in report["limitations"])
    lines += [
        "",
        "The original and added caches retain their image checksums and terms. The combined manifest, shard references, and scene registry are frozen beside the run artifacts. [Machine-readable summary](full_fivek_summary.json).",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("registry", type=Path)
    args = parser.parse_args()
    report = summarize(args.run_directory, args.registry)
    print(
        json.dumps(
            {
                "report": str(args.run_directory / "REPORT.md"),
                "cohorts": {
                    name: {
                        "scenes": group["scenes"],
                        "previous_psnr": group["models"]["previous"]["psnr"]["mean"],
                        "joint_psnr": group["models"]["joint"]["psnr"]["mean"],
                        "delta": group["joint_minus_previous"]["psnr"],
                    }
                    for name, group in report["cohorts"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
