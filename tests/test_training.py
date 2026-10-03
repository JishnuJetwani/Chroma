"""Small CPU integration checks; these establish mechanics, not photo quality."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from photograder.data import RENDER_VERSION, sha256_file, write_manifest
from photograder.models import ModelConfig, build_model
from photograder.recipes import EditRecipe
from photograder.runtime import load_checkpoint, save_checkpoint
from photograder.training import (
    DEFAULTS,
    LOSS_DEFAULTS,
    _loss,
    _resolve,
    exposure_auxiliary_loss,
    masked_mean,
    train,
    validation_loss,
)


@pytest.fixture
def paired_manifest(tmp_path):
    rng = np.random.default_rng(88)
    rows = []
    for index, split in enumerate(("train", "train", "val", "test")):
        pixels = rng.integers(40, 210, size=(24, 32, 3), dtype=np.uint8)
        source, target = tmp_path / f"{index}-in.png", tmp_path / f"{index}-target.png"
        Image.fromarray(pixels).save(source)
        Image.fromarray(np.rint(pixels.astype(float) * 0.7).astype(np.uint8)).save(target)
        rows.append(
            {
                "source_id": str(index),
                "scene_id": str(index),
                "input": source.name,
                "target": target.name,
                "style_id": 0,
                "style_name": "Expert C",
                "split": split,
                "render_version": RENDER_VERSION,
                "sha256": {"input": sha256_file(source), "target": sha256_file(target)},
            }
        )
    path = tmp_path / "pairs.jsonl"
    write_manifest(rows, path)
    return path


def test_checkpoint_roundtrip_includes_trained_status_and_style_order(tmp_path):
    model = build_model(ModelConfig(architecture="fixed", lut_size=5, styles=("Expert C", "Expert A")))
    with torch.no_grad():
        model.bank.tables[0].mul_(0.8)
    path = tmp_path / "roundtrip.pt"
    save_checkpoint(path, model, epoch=3, step=12, metadata={"trained": True, "pilot": True})
    restored, payload = load_checkpoint(path, "cpu")
    assert payload["metadata"]["trained"] is True
    assert payload["epoch"] == 3 and payload["step"] == 12
    assert restored.config.styles == ("Expert C", "Expert A")
    image = torch.rand(1, 3, 12, 15)
    torch.testing.assert_close(restored(image), model(image), atol=0, rtol=0)


def test_actual_cpu_training_reduces_loss_and_saves_loadable_checkpoint(paired_manifest, tmp_path):
    output = tmp_path / "training"
    config = {
        "model": {"architecture": "fixed", "lut_size": 5},
        "data": {"manifest": str(paired_manifest), "image_size": 32},
        "training": {
            "stage": "global",
            "device": "cpu",
            "epochs": 12,
            "batch_size": 2,
            "accumulation_steps": 1,
            "learning_rate": 0.02,
            "patience": 12,
        },
        "output_dir": str(output),
    }
    summary = train(config)
    history = json.loads((output / "history.json").read_text())
    assert min(row["validation_l1"] for row in history) < history[0]["validation_l1"] * 0.5
    assert summary["trained"] is True
    assert summary["total_optimizer_steps"] == 12
    model, payload = load_checkpoint(output / "best.pt")
    assert payload["metadata"]["trained"] is True
    assert torch.isfinite(model(torch.rand(1, 3, 17, 23))).all()


def test_repair_training_bypasses_global_branch_and_respects_padding():
    torch.set_num_threads(2)
    model = build_model(ModelConfig(preview_size=32, map_size=8, lut_size=5))
    image = torch.rand(1, 3, 24, 32) * 0.7 + 0.1
    mask = torch.ones(1, 1, 24, 32)
    mask[..., :4, :] = 0
    batch = {"image": image, "target": image * 0.9, "valid_mask": mask, "style_id": torch.tensor([0])}
    settings = {**DEFAULTS, "stage": "repair", "device": "cpu", "seed": 42}
    loss, terms = _loss(model, batch, settings, LOSS_DEFAULTS, 1)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.bank.tables.grad is None
    assert model.mixer.weight.grad is None
    assert model.decoder.output.weight.grad is not None
    assert "preservation" in terms
    values = torch.zeros_like(image)
    values[..., :4, :] = 1000
    assert masked_mean(values, mask) == 0


def test_resume_rejects_model_configuration_change(paired_manifest, tmp_path):
    model = build_model(ModelConfig(architecture="fixed", lut_size=5))
    checkpoint = tmp_path / "resume" / "last.pt"
    save_checkpoint(checkpoint, model, metadata={"trained": True})
    config = {
        "model": {"architecture": "fixed", "lut_size": 7},
        "data": {"manifest": str(paired_manifest), "image_size": 32},
        "training": {"device": "cpu"},
        "output_dir": str(tmp_path / "resume"),
    }
    with pytest.raises(ValueError, match="configuration differs"):
        train(config, resume=checkpoint)


def test_fresh_training_never_overwrites_existing_run(paired_manifest, tmp_path):
    output = tmp_path / "existing-run"
    output.mkdir()
    sentinel = output / "config.json"
    sentinel.write_text('{"existing":true}')
    config = {
        "model": {"architecture": "fixed", "lut_size": 5},
        "data": {"manifest": str(paired_manifest), "image_size": 32},
        "training": {"device": "cpu"},
        "output_dir": str(output),
    }
    with pytest.raises(FileExistsError, match="already exists"):
        train(config)
    assert sentinel.read_text() == '{"existing":true}'


def test_style_labels_must_match_manifest_indices(paired_manifest, tmp_path):
    config = {
        "model": {"architecture": "fixed", "lut_size": 5, "styles": ["Wrong editor"]},
        "data": {"manifest": str(paired_manifest), "image_size": 32},
        "training": {"device": "cpu"},
        "output_dir": str(tmp_path / "wrong-style"),
    }
    with pytest.raises(ValueError, match="does not match model.styles"):
        train(config)


def test_resume_cannot_silently_change_training_stage(paired_manifest, tmp_path):
    from photograder.corruptions import CORRUPTION_VERSION
    from photograder.runtime import file_sha256

    output = tmp_path / "resume-stage"
    config = {
        "model": {"architecture": "fixed", "lut_size": 5},
        "data": {"manifest": str(paired_manifest), "image_size": 32},
        "training": {"device": "cpu", "stage": "global", "epochs": 3},
        "output_dir": str(output),
    }
    checkpoint = output / "last.pt"
    save_checkpoint(
        checkpoint,
        build_model(ModelConfig(architecture="fixed", lut_size=5)),
        epoch=1,
        step=1,
        training_config=config,
        metadata={
            "trained": True,
            "manifest_sha256": file_sha256(paired_manifest),
            "corruption_version": CORRUPTION_VERSION,
        },
    )
    config["training"]["stage"] = "repair"
    with pytest.raises(ValueError, match="Cannot change stage"):
        train(config, resume=checkpoint)


def test_exposure_auxiliary_supervises_only_known_inverses_and_valid_pixels():
    image = torch.full((2, 3, 20, 28), 0.4)
    exposure = torch.zeros(2, 1, 5, 7, requires_grad=True)
    recipe = EditRecipe(exposure=exposure, metadata={"upsampling": "bilinear"})
    valid = torch.ones(2, 1, 20, 28)
    valid[..., :4, :] = 0
    inverse = torch.full_like(valid, -0.3)
    inverse[..., :4, :] = 900  # excluded, even with an extreme bogus label
    corruption = {"inverse_known": torch.tensor([True, False]), "known_inverse": inverse}
    loss = exposure_auxiliary_loss(recipe, image, valid, corruption)
    assert loss.item() == pytest.approx(0.3, abs=1e-6)
    loss.backward()
    assert exposure.grad[0].abs().sum() > 0
    assert torch.count_nonzero(exposure.grad[1]) == 0
    corruption["inverse_known"][:] = False
    assert exposure_auxiliary_loss(recipe, image, valid, corruption) == 0
    assert exposure_auxiliary_loss(recipe, image, valid, None) == 0


def test_repair_auxiliary_reaches_exposure_head_but_not_contrast_or_global(monkeypatch):
    from photograder import corruptions
    from photograder.color import linear_to_srgb, srgb_to_linear

    model = build_model(ModelConfig(preview_size=32, map_size=8, lut_size=5))
    image = torch.full((1, 3, 24, 32), 0.4)
    valid = torch.ones(1, 1, 24, 32)

    def pure_exposure(clean, **kwargs):
        return {
            "image": linear_to_srgb(srgb_to_linear(clean) * (2**0.3)),
            "mask": valid,
            "inverse_known": torch.tensor([True]),
            "known_inverse": torch.full_like(valid, -0.3),
        }

    monkeypatch.setattr(corruptions, "corrupt_batch", pure_exposure)
    batch = {"image": image, "target": image, "valid_mask": valid, "style_id": torch.tensor([0])}
    settings = {**DEFAULTS, "stage": "repair", "device": "cpu"}
    weights = {key: 0.0 for key in LOSS_DEFAULTS}
    weights["exposure_aux"] = 1.0
    loss, terms = _loss(model, batch, settings, weights, 0)
    assert terms["exposure_aux"] > 0.29
    loss.backward()
    assert model.decoder.output.weight.grad[0].abs().sum() > 0
    assert torch.count_nonzero(model.decoder.output.weight.grad[1]) == 0
    assert model.bank.tables.grad is None
    assert model.mixer.weight.grad is None
    _, real_terms = _loss(model, batch, {**settings, "stage": "joint"}, weights, 0)
    assert real_terms["exposure_aux"] == 0


def test_combined_fault_curriculum_threshold_does_not_change_validation(monkeypatch):
    from photograder import corruptions

    captured = []

    def unchanged(clean, **kwargs):
        captured.append(kwargs["allow_combinations"])
        return {"image": clean, "mask": torch.zeros_like(clean[:, :1])}

    monkeypatch.setattr(corruptions, "corrupt_batch", unchanged)
    model = build_model(ModelConfig(preview_size=32, map_size=8, lut_size=5))
    image = torch.full((1, 3, 24, 32), 0.4)
    batch = {
        "image": image,
        "target": image,
        "valid_mask": torch.ones_like(image[:, :1]),
        "style_id": torch.tensor([0]),
    }
    settings = {**DEFAULTS, "stage": "repair", "combined_faults_after_micro_step": 8}
    _loss(model, batch, settings, LOSS_DEFAULTS, 7)
    _loss(model, batch, settings, LOSS_DEFAULTS, 8)
    validation_loss(model, [batch], torch.device("cpu"), "repair", 42)
    assert captured == [False, True, False]
    defaults = _resolve({})
    assert defaults["training"]["combined_faults_after_micro_step"] is None
    assert defaults["loss"]["exposure_aux"] == 0
    with pytest.raises(ValueError, match="nonnegative integer"):
        _resolve({"training": {"combined_faults_after_micro_step": -1}})
