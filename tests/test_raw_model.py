import io
from dataclasses import asdict

import pytest
import torch

from photograder.color import luminance
from photograder.raw_model import (
    RawEditRecipe,
    RawGradingModel,
    RawModelConfig,
    additive_linear_veil,
    apply_tone_curve,
    baseline_raw,
    encode_raw_preview,
    monotone_tone_knots,
    raw_supervised_loss,
    render_raw,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def model(**kwargs):
    return RawGradingModel(RawModelConfig(preview_size=64, map_size=16, **kwargs))


def test_neutral_development_keeps_scene_highlights_distinct():
    values = torch.tensor([0.0, 0.18, 1.0, 2.0, 4.0, 16.0])
    image = values[None, None, None].expand(1, 3, 1, -1)
    output = baseline_raw(image)
    assert bool((output[0, 0, 0].diff() > 0).all())
    assert output[0, 0, 0, 2].item() == pytest.approx(0.7353569, abs=1e-6)
    encoded = encode_raw_preview(torch.tensor([-2.0, 0.0, 1.0, 2.0, 16.0]))
    assert bool((encoded.diff() > 0).all())
    assert encoded[-1] > 1


def test_model_neutral_initialization_parameter_budget_and_backward():
    network = model(styles=("C", "B"))
    assert sum(parameter.numel() for parameter in network.parameters()) < 2_000_000
    image = torch.rand(2, 3, 37, 61) * 5 - 0.1
    recipe = network.analyze(image, torch.tensor([0, 1]))
    assert recipe.local_exposure_ev.shape == (2, 1, 10, 16)
    assert recipe.source_size == (37, 61)
    prediction = render_raw(image, recipe)
    torch.testing.assert_close(prediction, baseline_raw(image), atol=2e-6, rtol=2e-5)
    loss = raw_supervised_loss(prediction, baseline_raw(image * 0.6), recipe)
    loss["total"].backward()
    gradients = [parameter.grad for parameter in network.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert network.global_head[-1].weight.grad.abs().sum() > 0
    assert network.local_head[-1].weight.grad.abs().sum() > 0


def test_curve_monotonic_even_with_adversarial_increment_logits():
    logits = torch.linspace(-100, 100, 32)[None].repeat(2, 1).requires_grad_()
    knots = monotone_tone_knots(logits)
    assert bool((knots.diff(dim=-1) > 0).all())
    torch.testing.assert_close(knots[:, 0], torch.zeros(2))
    torch.testing.assert_close(knots[:, -1], torch.ones(2))
    coordinate = torch.linspace(0, 1, 8193)[None, None, None].repeat(2, 1, 1, 1)
    output = apply_tone_curve(coordinate, knots)
    assert bool((output.diff(dim=-1) >= 0).all())
    output.mean().backward()
    assert torch.isfinite(logits.grad).all()


def test_fixed_global_recipe_monotonic_above_white():
    network = model()
    image = torch.linspace(0, 16, 8193)[None, None, None].expand(1, 3, 1, -1)
    recipe = network.analyze(image)
    recipe.tone_y = monotone_tone_knots(torch.randn(1, 32, generator=torch.Generator().manual_seed(7)) * 3)
    recipe.exposure_ev = torch.tensor([[1.5]])
    output = render_raw(image, recipe)
    assert bool((luminance(output).diff(dim=-1) >= 0).all())
    assert output[..., -1].mean() > output[..., 1024].mean()


def test_nontrivial_recipe_tiling_and_cache_roundtrip():
    torch.manual_seed(42)
    network = model(enable_veil=True)
    with torch.no_grad():
        network.global_head[-1].weight.normal_(std=0.03)
        network.local_head[-1].weight.normal_(std=0.03)
    image = torch.rand(1, 3, 73, 117) * 4 - 0.05
    recipe = network.analyze(image).detach()
    file = io.BytesIO()
    torch.save({"config": asdict(network.config), "recipe": recipe.to("cpu").state_dict()}, file)
    file.seek(0)
    state = torch.load(file, weights_only=True)
    restored = RawEditRecipe.from_state_dict(state["recipe"])
    assert RawModelConfig(**state["config"]) == network.config
    whole, whole_aux = render_raw(image, restored, strength=0.75, return_aux=True, exposure_bias_ev=-0.5)
    tiled, tiled_aux = render_raw(
        image, restored, strength=0.75, tile_size=23, return_aux=True, exposure_bias_ev=-0.5
    )
    torch.testing.assert_close(tiled, whole, atol=2e-6, rtol=2e-5)
    for key in whole_aux:
        torch.testing.assert_close(tiled_aux[key], whole_aux[key], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(render_raw(image, restored, strength=0), baseline_raw(image), atol=0, rtol=0)
    torch.testing.assert_close(
        render_raw(image, restored, strength=0, exposure_bias_ev=1), baseline_raw(image * 2), atol=0, rtol=0
    )


@pytest.mark.parametrize("value", [0.0, -2.0, 100.0])
def test_zero_negative_and_bright_inputs_remain_finite(value):
    network = model(enable_veil=True)
    image = torch.full((1, 3, 35, 59), value)
    recipe = network.analyze(image)
    output = render_raw(image, recipe)
    assert torch.isfinite(output).all()
    assert output.amin() >= 0 and output.amax() <= 1
    assert all(
        torch.isfinite(value).all()
        for value in recipe.state_dict().values()
        if isinstance(value, torch.Tensor)
    )
    output.mean().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
        if parameter.grad is not None
    )


def test_learned_color_matrix_preserves_luminance_and_gray():
    network = model()
    with torch.no_grad():
        network.global_head[-1].bias[4:13].normal_()
    recipe = network.analyze(torch.rand(1, 3, 32, 48))
    luma = torch.tensor([0.2126, 0.7152, 0.0722])[None, None]
    torch.testing.assert_close(luma @ recipe.color_matrix, luma)
    torch.testing.assert_close(recipe.color_matrix @ torch.ones(1, 3, 1), torch.ones(1, 3, 1))


def test_loss_excludes_clipped_pixels_in_reconstruction_and_gradients():
    network = model()
    image = torch.rand(1, 3, 16, 24)
    recipe = network.analyze(image)
    target = torch.rand_like(image)
    prediction = target.clone()
    prediction[..., 4:8, 5:9] = 0
    prediction.requires_grad_()
    available = torch.ones_like(image[:, :1])
    available[..., 4:8, 5:9] = 0
    losses = raw_supervised_loss(prediction, target, recipe, available_mask=available)
    assert losses["rgb_l1"] == 0
    assert losses["gradient"] == 0
    assert losses["highlight"] == 0
    losses["total"].backward()
    assert torch.count_nonzero(prediction.grad) == 0


def test_veil_does_not_assume_linear_one_is_sensor_clipping():
    image = torch.tensor([[[[0.5, 2.0, 0.7]], [[0.3, 0.2, 0.1]], [[0.2, 0.1, 0.1]]]])
    veil = torch.tensor([[[[0.0, 0.5, 0.5]], [[0.0, 0.2, 0.5]], [[0.0, 0.1, 0.5]]]])
    result = additive_linear_veil(image, veil)
    torch.testing.assert_close(result["image"], image + veil)
    assert not result["clipped"].any()
    assert result["available"].all()
    torch.testing.assert_close(result["image"][..., 0], image[..., 0], atol=0, rtol=0)
    clipped = additive_linear_veil(image, veil, clip_level=1)
    assert clipped["clipped"][0, 0, 0, 2]
    assert clipped["available"][0, 0, 0, 2] == 0
    assert not clipped["clipped"][0, 1, 0, 2]


def test_masks_and_style_errors():
    network = model()
    image = torch.rand(1, 3, 32, 48)
    with pytest.raises(ValueError, match="style index"):
        network.analyze(image, style=3)
    with pytest.raises(ValueError, match="valid_mask"):
        network.analyze(image, valid_mask=torch.ones_like(image))
    recipe = network.analyze(image, valid_mask=torch.zeros_like(image[:, :1]))
    assert torch.isfinite(render_raw(image, recipe)).all()
    with pytest.raises(ValueError, match="strength"):
        render_raw(image, recipe, strength=1.1)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_raw_mps_forward_backward_matches_cpu():
    torch.manual_seed(53)
    cpu = model(enable_veil=True)
    with torch.no_grad():
        cpu.global_head[-1].weight.normal_(std=0.01)
        cpu.local_head[-1].weight.normal_(std=0.01)
    metal = model(enable_veil=True).to("mps")
    metal.load_state_dict(cpu.state_dict())
    image = torch.rand(1, 3, 33, 49) * 3 - 0.03
    expected, actual = cpu(image), metal(image.to("mps"))
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-3)
    expected.mean().backward()
    actual.mean().backward()
    for left, right in zip(cpu.parameters(), metal.parameters(), strict=False):
        if left.grad is not None:
            assert torch.isfinite(right.grad).all()
            torch.testing.assert_close(right.grad.cpu(), left.grad, atol=1e-4, rtol=0.04)
