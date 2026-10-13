"""Summarize the frozen holdout after training, without selecting checkpoints."""

import argparse
import csv
import json
from pathlib import Path

from photograder.metrics import scene_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory
    run = json.loads((root / "run.json").read_text())
    if run["status"] != "completed":
        raise ValueError("Only report a completed, evaluated curriculum")
    records = {}
    for stage in ("global", "joint"):
        with (root / "evaluation" / stage / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        records[stage] = {(row["scene_id"], int(row["style_id"])): row for row in rows}
        if len(rows) != len(records[stage]):
            raise ValueError("Duplicate scene/style observations")
    if records["global"].keys() != records["joint"].keys():
        raise ValueError("Model reports do not describe the same held-out pairs")
    report = {"run": str(root), "primary_style": "FiveK Expert C", "styles": {}}
    for style, name in enumerate(("C", "A", "B")):
        keys = [key for key in records["joint"] if key[1] == style]
        scenes = [key[0] for key in keys]
        group = {"scenes": len(set(scenes)), "models": {}, "paired_joint_minus_global": {}}
        for model, stage, prefix in (
            ("input", "joint", "input_"),
            ("global", "global", ""),
            ("joint", "joint", ""),
        ):
            group["models"][model] = {
                metric: scene_bootstrap(
                    [float(records[stage][key][prefix + metric]) for key in keys], scenes, seed=run["seed"]
                )
                for metric in ("psnr", "ssim", "delta_e76", "mae")
            }
        for metric in ("psnr", "ssim", "new_clipped_high_fraction", "all_channels_white_fraction"):
            group["paired_joint_minus_global"][metric] = scene_bootstrap(
                [
                    float(records["joint"][key][metric]) - float(records["global"][key][metric])
                    for key in keys
                ],
                scenes,
                seed=run["seed"],
            )
        report["styles"][name] = group
    (root / "holdout_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    primary = report["styles"]["C"]
    lines = [
        "# Fresh full-data 70/30 run",
        "",
        f"Completed on **{run['gpu']}**. Fixed schedule: 10 global + 5 repair + 10 joint epochs.",
        "",
        "1,924 training and 824 held-out scenes; 1,958/838 editor pairs. No test-based selection. All stages started from the same fresh curriculum, with no earlier pilot weights.",
        "",
        "## Primary Expert C holdout (824 photos)",
        "",
        "| Output | PSNR ↑ | SSIM ↑ | CIE76 color error ↓ |",
        "|---|---:|---:|---:|",
    ]
    for name in ("input", "global", "joint"):
        values = primary["models"][name]
        lines.append(
            f"| {name.title()} | {values['psnr']['mean']:.3f} dB | {values['ssim']['mean']:.4f} | {values['delta_e76']['mean']:.3f} |"
        )
    difference = primary["paired_joint_minus_global"]["psnr"]
    clipping = primary["paired_joint_minus_global"]["new_clipped_high_fraction"]
    lines += [
        "",
        f"Joint minus global PSNR: **{difference['mean']:+.3f} dB**, paired scene-bootstrap 95% CI [{difference['ci_low']:+.3f}, {difference['ci_high']:+.3f}].",
        "",
        f"Newly clipped highlight pixels increase by **{100 * clipping['mean']:.3f} percentage points** versus global-only (95% CI [{100 * clipping['ci_low']:.3f}, {100 * clipping['ci_high']:.3f}]). This is a measurable tradeoff requiring visual inspection; clipping alone does not prove an unwanted edit.",
        "",
        "SSIM is a structural similarity index, not a percentage of photographic quality or an accuracy score. The global and joint checkpoints have different total training budgets.",
        "",
        "## Runtime and cost",
        "",
        f"Recorded GPU function work across attempts: {run['elapsed_seconds'] / 60:.2f} minutes. Estimated function compute: ${run['estimated_compute_usd']:.3f}; this is not the account bill and excludes preparation, startup/idle, any unrecorded cancellation tail, storage and credit adjustments.",
        f"Selected microbatch {run['selected']['batch_size']}, accumulation {run['selected']['accumulation_steps']}, CNN precision {run['selected']['encoder_autocast']}; renderer FP32.",
        "",
        "## Limits",
        "",
        "The custom split differs from the published FiveK split and earlier pilot holdout, so do not directly compare their means. Metrics use padded 256px images with valid-content masks. No convergence claim is made after a fixed schedule. A/B each have only seven test scenes and 17 training scenes. This run does not establish real local defect detection, preservation-training causality, halo-free full-resolution rendering, or blind preference wins.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(root / "REPORT.md"), "expert_c": primary}, indent=2))


if __name__ == "__main__":
    main()
