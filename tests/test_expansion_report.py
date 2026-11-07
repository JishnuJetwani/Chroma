import csv
import importlib.util
import json
from pathlib import Path

import pytest


def test_report_uses_same_photos_and_separates_new_holdout(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "report_expansion", Path(__file__).parents[1] / "scripts/report_fivek_expansion.py"
    )
    report_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report_module)
    scenes = [f"fivek:example{i:04d}" for i in range(1500)]
    run = {
        "status": "completed",
        "manifest_sha256": "frozen",
        "seed": 42,
        "scene_counts": {"train": 3500, "test": 1500},
        "pair_counts": {"train": 3534, "val": 0, "test": 1514},
        "schedule": {"global": 10, "repair": 5, "joint": 10},
        "gpu": "test",
        "comparison_checkpoint_sha256": "previous",
        "elapsed_seconds": 60,
        "estimated_compute_usd": 0.1,
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    registry = {
        "original_manifest_sha256": "frozen",
        "outer_splits": {"test": scenes},
        "cohorts": {"original": scenes[:824], "additional": scenes[824:]},
    }
    registry_path = tmp_path / "split.json"
    registry_path.write_text(json.dumps(registry))
    for model, psnr in (("previous", 22), ("global", 23), ("joint", 24)):
        destination = tmp_path / "evaluation" / model / "metrics.csv"
        destination.parent.mkdir(parents=True)
        rows = [
            {
                "scene_id": scene,
                "style_id": 0,
                "psnr": psnr,
                "ssim": 0.9,
                "delta_e76": 8,
                "new_clipped_high_fraction": 0.01,
                "new_clipped_low_fraction": 0.01,
                "input_psnr": 18,
                "input_ssim": 0.8,
                "input_delta_e76": 16,
            }
            for scene in scenes
        ]
        with destination.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(reversed(rows))  # Match by scene identity, never CSV position.
    report = report_module.summarize(tmp_path, registry_path)
    assert report["cohorts"]["original"]["scenes"] == 824
    assert report["cohorts"]["additional"]["scenes"] == 676
    for group in report["cohorts"].values():
        assert group["joint_minus_previous"]["psnr"]["mean"] == 2
        assert group["joint_minus_previous"]["psnr"]["ci_low"] == 2
        assert group["psnr_wins"] == group["scenes"]
    broken = tmp_path / "evaluation/previous/metrics.csv"
    broken.write_text("\n".join(broken.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="every held-out"):
        report_module.summarize(tmp_path, registry_path)
