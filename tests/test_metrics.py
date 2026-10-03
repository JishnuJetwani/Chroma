import json

import pytest
import torch

from photograder.evaluation import evaluate
from photograder.metrics import (
    delta_e76,
    masked_mean,
    per_image_metrics,
    psnr,
    scene_bootstrap,
    srgb_to_lab,
    ssim,
)


def test_known_psnr_and_mask_exclusion():
    target = torch.zeros(1, 3, 12, 12)
    output = target + 0.1
    assert psnr(output, target).item() == pytest.approx(20, abs=0.0001)
    output[:, :, :6] = 0.5
    mask = torch.zeros(1, 1, 12, 12)
    mask[:, :, 6:] = 1
    assert psnr(output, target, mask).item() == pytest.approx(20, abs=0.0001)
    assert torch.isnan(psnr(output, target, torch.zeros_like(mask))).all()


def test_perfect_match_metrics_and_ssim_small_image():
    image = torch.rand(2, 3, 9, 13)
    values = per_image_metrics(image, image)
    assert torch.equal(values["mae"], torch.zeros(2))
    assert torch.equal(values["delta_e76"], torch.zeros(2))
    assert torch.allclose(values["ssim"], torch.ones(2), atol=1e-06)
    assert torch.allclose(values["psnr"], torch.full((2,), 120.0))
    tiny = torch.rand(1, 3, 1, 1)
    assert ssim(tiny, tiny).item() == pytest.approx(1)


def test_ssim_is_invariant_to_changes_inside_privacy_and_padding_masks():
    torch.manual_seed(11)
    target = torch.rand(1, 3, 64, 80)
    output = (target + 0.03).clamp(0, 1)
    valid = torch.ones(1, 1, 64, 80)
    valid[..., :8, :] = 0
    valid[..., -8:, :] = 0
    valid[..., 24:40, 32:48] = 0
    expected = ssim(output, target, valid)
    changed = torch.where(valid.bool(), output, torch.zeros_like(output))
    torch.testing.assert_close(ssim(changed, target, valid), expected, atol=0, rtol=0)
    assert ssim(changed, target).item() < ssim(output, target).item()


def test_ssim_omits_examples_with_no_fully_valid_window():
    image = torch.rand(1, 3, 20, 20)
    valid = torch.zeros(1, 1, 20, 20)
    valid[..., 8:12, 8:12] = 1
    assert torch.isnan(ssim(image, image, valid)).all()
    torch.testing.assert_close(ssim(image, image, torch.ones_like(valid)), ssim(image, image))


def test_lab_white_black_and_delta_e_units():
    white = torch.ones(1, 3, 4, 4)
    black = torch.zeros_like(white)
    lab = srgb_to_lab(white)
    assert lab[0, 0, 0, 0].item() == pytest.approx(100, abs=0.001)
    assert lab[0, 1:, 0, 0].abs().max().item() < 0.001
    assert delta_e76(white, black).item() == pytest.approx(100, abs=0.001)


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
