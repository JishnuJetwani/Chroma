import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from photograder.inference import Grader, image_tensor, tensor_image
from photograder.models import ModelConfig, build_model
from photograder.recipes import EditRecipe
from photograder.renderer import render
from photograder.runtime import save_checkpoint


@pytest.fixture
def grader(tmp_path):
    model = build_model(ModelConfig(architecture="fixed", lut_size=5))
    with torch.no_grad():
        model.bank.tables.mul_(0.75)
    checkpoint = tmp_path / "trained.pt"
    save_checkpoint(checkpoint, model, metadata={"trained": True, "pilot": True})
    return Grader(checkpoint, device="cpu")


def test_untrained_checkpoints_cannot_be_presented_as_grading_models(tmp_path):
    checkpoint = tmp_path / "untrained.pt"
    save_checkpoint(checkpoint, build_model(ModelConfig(architecture="fixed", lut_size=5)))
    with pytest.raises(ValueError, match="not been trained"):
        Grader(checkpoint, device="cpu")


def test_export_zero_strength_preserves_pixels_dimensions_and_original(grader, tmp_path):
    pixels = np.random.default_rng(16).integers(0, 256, size=(39, 67, 3), dtype=np.uint8)
    source = tmp_path / "original.png"
    Image.fromarray(pixels).save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    analysis = grader.analyze(source)
    result = grader.export(analysis, tmp_path / "bypass.png", 0.0, 0.0)
    with Image.open(result["path"]) as exported:
        assert exported.size == (67, 39)
        np.testing.assert_array_equal(np.asarray(exported), pixels)
        assert exported.info.get("icc_profile")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="original is preserved"):
        grader.export(analysis, source)
    with pytest.raises(FileExistsError):
        grader.export(analysis, result["path"])


def test_cached_recipe_rerender_never_reanalyzes_and_can_be_saved(grader, tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.fromarray(np.full((24, 33, 3), 160, dtype=np.uint8)).save(source)
    analysis = grader.analyze(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("Slider rendering must not run model analysis")

    monkeypatch.setattr(grader.model, "analyze", forbidden)
    full = np.asarray(grader.render_preview(analysis, 1.0, 0.0))
    half = np.asarray(grader.render_preview(analysis, 0.5, 0.0))
    assert full.mean() < half.mean() < 160
    recipe_path = tmp_path / "recipe.pt"
    grader.save_recipe(analysis, recipe_path)
    payload = torch.load(recipe_path, weights_only=True)
    recipe = EditRecipe.from_state_dict(payload["recipe"])
    torch.testing.assert_close(recipe.luts, analysis.recipe.luts)
    assert payload["source_dimensions"] == [33, 24]
    assert payload["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        grader.save_recipe(analysis, source)
    with pytest.raises(FileExistsError):
        grader.save_recipe(analysis, recipe_path)


def test_full_resolution_export_reuses_aspect_aligned_maps_beyond_preview(grader, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    source = tmp_path / "large-photo.png"
    Image.fromarray(np.full((768, 1536, 3), 96, dtype=np.uint8)).save(source)
    analysis = grader.analyze(source)
    assert analysis.preview.size == (1024, 512)
    # A known, asymmetric recipe exposes swapped axes or per-tile map restarts.
    analysis.recipe = EditRecipe(
        exposure=torch.linspace(-0.6, 0.6, 16).view(1, 1, 1, 16).expand(1, 1, 8, 16),
        log_contrast=torch.full((1, 1, 8, 16), 0.4),
        source_size=(512, 1024),
        metadata={"upsampling": "bilinear"},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Export must reuse the original analysis recipe")

    monkeypatch.setattr(grader.model, "analyze", forbidden)
    with torch.inference_mode():
        expected = np.asarray(tensor_image(render(image_tensor(analysis.original), analysis.recipe)))
    result = grader.export(analysis, tmp_path / "full-size.png", 1.0, 1.0)
    with Image.open(result["path"]) as exported:
        actual = np.asarray(exported)
        assert exported.size == (1536, 768)
    assert abs(actual.astype(int) - expected.astype(int)).max() <= 1
    assert actual[:, :128].mean() < 96 < actual[:, -128:].mean()
    # Full-size coordinates keep the ramp monotonic across both 512px seams.
    assert np.diff(actual.mean(axis=(0, 2))).min() >= -1


def test_transparent_inputs_fail_explicitly(grader, tmp_path):
    source = tmp_path / "transparent.png"
    image = np.full((12, 15, 4), 100, dtype=np.uint8)
    image[..., 3] = 255
    image[0, 0, 3] = 0
    Image.fromarray(image).save(source)
    with pytest.raises(ValueError, match="transparency"):
        grader.analyze(source)
