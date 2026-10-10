import csv
import json

import pytest
import torch

from photograder.evaluation import evaluate
from photograder.experiments import analyze_votes, compare_evaluations, create_blind_comparisons, run_matrix
from photograder.metrics import (
    clipping_metrics,
    delta_e76,
    locality_metrics,
    masked_mean,
    noop_metrics,
    per_image_metrics,
    psnr,
    scene_bootstrap,
    srgb_to_lab,
    ssim,
)


def test_known_psnr_and_mask_exclusion():
    target = torch.zeros(1, 3, 12, 12)
    output = target + 0.1
    assert psnr(output, target).item() == pytest.approx(20, abs=1e-4)
    output[:, :, :6] = 0.5
    mask = torch.zeros(1, 1, 12, 12)
    mask[:, :, 6:] = 1
    assert psnr(output, target, mask).item() == pytest.approx(20, abs=1e-4)
    assert torch.isnan(psnr(output, target, torch.zeros_like(mask))).all()


def test_perfect_match_metrics_and_ssim_small_image():
    image = torch.rand(2, 3, 9, 13)
    values = per_image_metrics(image, image)
    assert torch.equal(values["mae"], torch.zeros(2))
    assert torch.equal(values["delta_e76"], torch.zeros(2))
    assert torch.allclose(values["ssim"], torch.ones(2), atol=1e-6)
    assert torch.allclose(values["psnr"], torch.full((2,), 120.0))
    tiny = torch.rand(1, 3, 1, 1)
    assert ssim(tiny, tiny).item() == pytest.approx(1)


def test_ssim_is_invariant_to_changes_inside_privacy_and_padding_masks():
    torch.manual_seed(11)
    target = torch.rand(1, 3, 64, 80)
    output = (target + 0.03).clamp(0, 1)
    valid = torch.ones(1, 1, 64, 80)
    valid[..., :8, :] = 0  # letterbox padding
    valid[..., -8:, :] = 0
    valid[..., 24:40, 32:48] = 0  # privacy region
    expected = ssim(output, target, valid)
    changed = torch.where(valid.bool(), output, torch.zeros_like(output))
    torch.testing.assert_close(ssim(changed, target, valid), expected, atol=0, rtol=0)
    # This is a real exclusion check: the unmasked score must see the damage.
    assert ssim(changed, target).item() < ssim(output, target).item()


def test_ssim_omits_examples_with_no_fully_valid_window():
    image = torch.rand(1, 3, 20, 20)
    valid = torch.zeros(1, 1, 20, 20)
    valid[..., 8:12, 8:12] = 1
    assert torch.isnan(ssim(image, image, valid)).all()
    # Full masks preserve normal reflect-padding behavior at real image edges.
    torch.testing.assert_close(ssim(image, image, torch.ones_like(valid)), ssim(image, image))


def test_lab_white_black_and_delta_e_units():
    white = torch.ones(1, 3, 4, 4)
    black = torch.zeros_like(white)
    lab = srgb_to_lab(white)
    assert lab[0, 0, 0, 0].item() == pytest.approx(100, abs=1e-3)
    assert lab[0, 1:, 0, 0].abs().max().item() < 1e-3
    assert delta_e76(white, black).item() == pytest.approx(100, abs=1e-3)


def test_local_recovery_uses_clean_model_output_for_outside_drift():
    clean = torch.full((1, 3, 24, 24), 0.4)
    mask = torch.zeros(1, 1, 24, 24)
    mask[:, :, 8:16, 8:16] = 1
    corrupted = clean + 0.2 * mask
    # The global grade legitimately shifts the whole photo, which should not be
    # counted as leakage outside the synthetic region.
    clean_graded = clean + 0.1
    corrected = clean_graded.clone()
    result = locality_metrics(clean_graded, corrected, corrupted + 0.1, clean_graded, mask, boundary_radius=2)
    assert result["outside_drift_mae"].item() == 0
    assert result["boundary_drift_mae"].item() == 0
    assert result["recovery_mae_gain"].item() == pytest.approx(0.2)
    wrong = corrected.clone()
    wrong[:, :, :2] += 0.1
    assert locality_metrics(clean_graded, wrong, corrupted, clean, mask, 2)["outside_drift_mae"].item() > 0
    valid = torch.ones_like(mask)
    valid[:, :, :2] = 0
    assert (
        locality_metrics(clean_graded, wrong, corrupted, clean, mask, 2, valid)["outside_drift_mae"].item()
        == 0
    )


def test_noop_and_new_clipping_are_explicit_diagnostics():
    reference = torch.full((1, 3, 4, 4), 0.5)
    assert noop_metrics(reference, reference)["noop_changed_fraction"].item() == 0
    output = reference.clone()
    output[:, 0, 0, 0] = 1
    result = clipping_metrics(output, reference)
    assert result["new_clipped_high_fraction"].item() == pytest.approx(1 / 16)
    assert result["all_channels_white_fraction"].item() == 0


def test_scene_bootstrap_does_not_count_duplicate_scene_as_independent():
    result = scene_bootstrap([0, 0, 1], ["a", "a", "b"], samples=100)
    assert result["mean"] == 0.5
    assert result["scenes"] == 2
    assert result["observations"] == 3
    assert result == scene_bootstrap([0, 0, 1], ["a", "a", "b"], samples=100)
    assert scene_bootstrap([float("nan")], ["a"])["mean"] is None


def test_invalid_inputs_are_rejected():
    image = torch.zeros(1, 3, 8, 8)
    with pytest.raises(ValueError):
        psnr(image, image[:, :, :-1])
    with pytest.raises(ValueError):
        masked_mean(image, torch.ones_like(image))
    with pytest.raises(ValueError):
        masked_mean(image, torch.full((1, 1, 8, 8), -1.0))


def test_evaluation_emits_portable_records_and_restores_training_mode(tmp_path):
    class Identity(torch.nn.Module):
        def forward(self, image, style):
            return image

    model = Identity().train()
    image = torch.rand(2, 3, 12, 12)
    batch = {
        "image": image,
        "target": image.clone(),
        "style_id": torch.zeros(2, dtype=torch.long),
        "scene_id": ["one", "two"],
        "valid_mask": torch.ones(2, 1, 12, 12),
    }
    result = evaluate(
        model,
        [batch],
        "cpu",
        tmp_path,
        checkpoint="identity-test",
        bootstrap_samples=10,
        save_images=True,
        protocol={"split": "synthetic_unit_test"},
    )
    assert model.training
    assert result["images"] == 2
    assert result["metrics"]["psnr"]["mean"] == 120
    assert json.loads((tmp_path / "metrics.json").read_text())["checkpoint"] == "identity-test"
    assert len(list((tmp_path / "comparisons").glob("*.png"))) == 2
    assert (tmp_path / "metrics.csv").exists()


def test_evaluation_controls_use_original_inputs_and_held_out_family(tmp_path, monkeypatch):
    from photograder import corruptions

    class Identity(torch.nn.Module):
        def forward(self, image, style):
            return image

    source = torch.full((1, 3, 20, 20), 0.3)
    target = source + 0.1
    valid = torch.ones(1, 1, 20, 20)
    valid[:, :, :2] = 0
    captured = {}

    def fake_corrupt(images, **kwargs):
        captured.update(kwargs)
        assert torch.equal(images, source)
        mask = torch.zeros_like(valid)
        mask[:, :, 8:12, 8:12] = 1
        return {"image": images + mask * 0.1, "mask": mask}

    monkeypatch.setattr(corruptions, "corrupt_batch", fake_corrupt)
    report = evaluate(
        Identity(),
        [
            {
                "image": source,
                "target": target,
                "valid_mask": valid,
                "scene_id": ["test"],
                "style_id": torch.zeros(1, dtype=torch.long),
            }
        ],
        "cpu",
        tmp_path,
        include_controls=True,
        bootstrap_samples=10,
    )
    assert captured["families"] == ("edge_aware",)
    assert captured["allow_combinations"]
    assert torch.equal(captured["valid_mask"], valid)
    assert report["metrics"]["control_outside_drift_mae"]["mean"] == 0
    assert report["controls"]["source"] == "original paired input"


def test_global_context_evaluation_receives_validity_mask(tmp_path):
    from types import SimpleNamespace

    image = torch.full((1, 3, 20, 20), 0.4)
    valid = torch.ones(1, 1, 20, 20)
    valid[:, :, :2] = 0

    class ContextIdentity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(global_context=True)
            self.calls = 0

        def forward(self, source, style, valid_mask=None):
            assert torch.equal(valid_mask, valid)
            self.calls += 1
            return source

    model = ContextIdentity()
    batch = {
        "image": image,
        "target": image,
        "valid_mask": valid,
        "scene_id": ["context-test"],
        "style_id": torch.zeros(1, dtype=torch.long),
    }
    evaluate(model, [batch], "cpu", tmp_path, include_controls=True, bootstrap_samples=10)
    assert model.calls == 2


def test_blind_comparisons_and_vote_analysis(tmp_path):
    from PIL import Image

    paths = []
    for index, value in enumerate((80, 90, 100)):
        path = tmp_path / f"image{index}.png"
        Image.new("RGB", (16, 16), (value, value, value)).save(path)
        paths.append(str(path))
    destination = tmp_path / "blind"
    create_blind_comparisons(
        [
            {
                "scene_id": "scene1",
                "image_id": "photo1",
                "input_path": paths[0],
                "a_path": paths[1],
                "b_path": paths[2],
            }
        ],
        destination,
    )
    with (destination / "private_key.csv").open() as handle:
        info = list(csv.DictReader(handle))[0]
    choice = "A" if info["left_model"] == "candidate" else "B"
    with (destination / "votes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["comparison_id", "rater_id", "overall", "local_correction", "fewer_artifacts"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "comparison_id": info["comparison_id"],
                "rater_id": "person1",
                "overall": choice,
                "local_correction": "tie",
                "fewer_artifacts": "",
            }
        )
    result = analyze_votes(destination / "votes.csv", destination / "private_key.csv", bootstrap_samples=10)
    assert result["outcomes"]["overall"]["mean"] == 1
    assert result["outcomes"]["local_correction"]["ties"] == 1
    assert result["outcomes"]["local_correction"]["mean"] is None


def test_preview_matrix_has_correct_independent_joint_initializations():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    planned = run_matrix(root / "configs/matrix_preview.json")
    assert [entry["id"] for entry in planned] == [
        "fixed_preview",
        "global_preview",
        "repair_preview",
        "local_preview",
        "local_no_preservation_preview",
    ]
    assert "--initialize" not in planned[0]["command"]
    assert "--initialize" not in planned[1]["command"]
    assert planned[2]["command"][-1] == str(root / "runs/preview_reproduction/global/best.pt")
    assert (
        planned[3]["command"][-1]
        == planned[4]["command"][-1]
        == str(root / "runs/preview_reproduction/repair/best.pt")
    )
    assert all(entry["working_directory"] == str(root) for entry in planned)
    assert all(not entry["executed"] for entry in planned)


def test_matched_comparison_uses_paired_deltas_and_rejects_different_protocol(tmp_path):
    reports = []
    for name, scores in (("reference", [20, 30]), ("candidate", [21, 31])):
        folder = tmp_path / name
        folder.mkdir()
        report = {
            "images": 2,
            "protocol": {"manifest_sha256": "same-data", "split": "test", "image_size": 256, "seed": 42},
            "metrics": {"psnr": {}},
            "metric_note": "fixed metric definition",
            "torch_version": "2.8.0",
            "device": "cpu",
            "controls": None,
            "model_config": {"styles": ["Expert C"]},
        }
        (folder / "metrics.json").write_text(json.dumps(report))
        with (folder / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scene_id", "style_id", "image_index", "psnr"])
            writer.writeheader()
            for index, score in enumerate(scores):
                writer.writerow({"scene_id": str(index), "style_id": 0, "image_index": index, "psnr": score})
        reports.append(str(folder))
    compared = compare_evaluations(reports, tmp_path / "comparison", bootstrap_samples=10)
    assert compared["comparisons"][0]["paired_deltas"]["psnr"]["mean"] == 1
    report["protocol"]["image_size"] = 512
    (folder / "metrics.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="image_size"):
        compare_evaluations(reports, tmp_path / "mismatch", bootstrap_samples=10)
