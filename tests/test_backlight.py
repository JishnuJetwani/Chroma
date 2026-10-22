import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from photograder.backlight import prepare_backlight_recipe
from photograder.inference import Grader, image_tensor
from photograder.models import ModelConfig, build_model
from photograder.recipes import EditRecipe
from photograder.renderer import identity_lut, render
from photograder.runtime import save_checkpoint


def test_policy_preserves_source_white_and_does_not_mutate_recipe():
    torch.set_num_threads(2)
    x = torch.linspace(0.1, 1.0, 161).view(1, 1, 1, -1).expand(1, 3, 53, 161).clone()
    x[..., -30:] = 1.0
    recipe = EditRecipe(
        luts=identity_lut(9)[None] * 0.8,
        exposure=torch.linspace(-0.9, 0.3, 16).reshape(1, 1, 4, 4),
        log_contrast=torch.ones(1, 1, 4, 4),
        metadata={"upsampling": "bilinear"},
    )
    before = {k: v.clone() for k, v in recipe.state_dict().items() if isinstance(v, torch.Tensor)}
    safe = prepare_backlight_recipe(recipe, x)
    y = render(x, safe)
    assert torch.equal(y[..., -30:], x[..., -30:])
    assert torch.isfinite(y).all()
    assert y.min() >= 0 and y.max() <= 1
    assert "backlight" not in recipe.metadata
    for key, value in before.items():
        assert torch.equal(value, getattr(recipe, key))
    assert torch.equal(render(x, safe, 0, 0), x)
    # Soft toe leaves recoverable positive shadow values above zero.
    assert (y[..., :10] > 0).all()


def test_backlight_export_tiles_share_policy_and_coordinates():
    torch.manual_seed(45)
    x = torch.rand(1, 3, 89, 137) * 0.7 + 0.2
    recipe = EditRecipe(
        luts=identity_lut(9)[None],
        exposure=torch.rand(1, 1, 8, 12) * 0.5 - 0.3,
        log_contrast=torch.rand(1, 1, 8, 12) * 0.3,
        guide=x.mean(1, keepdim=True),
    )
    safe = prepare_backlight_recipe(recipe, x)
    torch.testing.assert_close(render(x, safe), render(x, safe, tile_size=41), atol=7e-5, rtol=0)


def test_backlight_cached_recipe_and_export_consistency(tmp_path, monkeypatch):
    model = build_model(ModelConfig(architecture="fixed", lut_size=9))
    with torch.no_grad():
        model.bank.tables.mul_(0.85)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, metadata={"trained": True})
    grader = Grader(checkpoint, "cpu")
    source = tmp_path / "source.png"
    pixels = np.random.default_rng(9).integers(40, 256, (63, 91, 3), dtype=np.uint8)
    pixels[:20, :30] = 255
    Image.fromarray(pixels).save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    analysis = grader.analyze(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("Slider changes/export may not rerun the CNN")

    monkeypatch.setattr(grader.model, "analyze", forbidden)
    preview = grader.render_preview(analysis, treatment="backlight")
    cached = analysis.backlight_recipe
    exported = grader.export(analysis, tmp_path / "graded.png", treatment="backlight")
    assert analysis.backlight_recipe is cached
    np.testing.assert_array_equal(np.asarray(preview), np.asarray(Image.open(exported["path"])))
    grader.export(analysis, tmp_path / "zero.png", 0, 0, treatment="backlight")
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "zero.png")), pixels)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    grader.save_recipe(analysis, tmp_path / "recipe.pt", treatment="backlight")
    saved = torch.load(tmp_path / "recipe.pt", weights_only=True)
    recipe = EditRecipe.from_state_dict(saved["recipe"])
    torch.testing.assert_close(
        render(image_tensor(analysis.original), recipe), render(image_tensor(analysis.original), cached)
    )
    with pytest.raises(ValueError, match="Unknown treatment"):
        grader.render_preview(analysis, treatment="invented")


@pytest.mark.mps
def test_backlight_cpu_mps_render_parity():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.manual_seed(53)
    x = torch.rand(1, 3, 49, 73) * 0.8 + 0.15
    recipe = EditRecipe(
        luts=identity_lut(9)[None] * 0.94,
        exposure=torch.rand(1, 1, 6, 9) * 0.3,
        guide=x.mean(1, keepdim=True),
    )
    safe = prepare_backlight_recipe(recipe, x)
    expected = render(x, safe, tile_size=31)
    actual = render(x.to("mps"), safe.to("mps"), tile_size=31).cpu()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
