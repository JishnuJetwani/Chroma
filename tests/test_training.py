"""Small CPU integration checks; these establish mechanics, not photo quality."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from photograder.data import RENDER_VERSION, sha256_file, write_manifest
from photograder.models import ModelConfig, build_model
from photograder.runtime import load_checkpoint, save_checkpoint
from photograder.training import (
    train,
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
