import pytest
import torch

from photograder.color import linear_to_srgb, luminance, srgb_to_linear
from photograder.recipes import EditRecipe
from photograder.renderer import apply_lut, exposure_field, identity_lut, render, render_with_aux


def test_color_roundtrip_and_finite_gradient_at_black():
    x = torch.tensor([0.0, 0.001, 0.04045, 0.5, 1.0], requires_grad=True)
    recovered = linear_to_srgb(srgb_to_linear(x))
    torch.testing.assert_close(recovered, x, atol=2e-7, rtol=1e-5)
    recovered.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_lut_axes_bounds_and_interpolation():
    x = torch.rand(2, 3, 11, 17)
    x[..., 0, 0] = 0
    x[..., -1, -1] = 1
    table = identity_lut(9)
    torch.testing.assert_close(apply_lut(x, table), x, atol=2e-7, rtol=1e-6)
    transformed = table[[2, 0, 1]] * 0.6 + 0.1
    torch.testing.assert_close(apply_lut(x, transformed), x[:, [2, 0, 1]] * 0.6 + 0.1, atol=2e-7, rtol=1e-6)


def test_lut_gradients_pass_numerical_check():
    torch.manual_seed(4)
    x = (torch.rand(1, 3, 2, 2, dtype=torch.float64) * 0.7 + 0.13).requires_grad_()
    lut = identity_lut(3, dtype=torch.float64)[None].requires_grad_()
    assert torch.autograd.gradcheck(apply_lut, (x, lut), eps=1e-6, atol=1e-5)


def test_exact_bypass_and_independent_strengths():
    x = torch.rand(1, 3, 30, 41) * 0.7 + 0.1
    lut = identity_lut(9)[None] * 0.8
    recipe = EditRecipe(
        luts=lut,
        exposure=torch.ones(1, 1, 8, 8) * 0.5,
        log_contrast=torch.zeros(1, 1, 8, 8),
        metadata={"upsampling": "bilinear"},
    )
    assert render(x, recipe, 0, 0) is x
    torch.testing.assert_close(render(x, recipe, 1, 0), apply_lut(x, lut))
    assert render(x, recipe, 0, 1).mean() > x.mean()
    recipe.exposure = -recipe.exposure
    assert render(x, recipe, 0, 1).mean() < x.mean()


def test_local_contrast_changes_detail_without_constant_region_shift():
    x = torch.full((1, 3, 32, 40), 0.4)
    recipe = EditRecipe(log_contrast=torch.full((1, 1, 8, 10), -1.0))
    torch.testing.assert_close(render(x, recipe), x, atol=1e-5, rtol=1e-5)
    x[..., ::2, ::2] += 0.02
    lowered = render(x, recipe)
    recipe.log_contrast *= -1
    raised = render(x, recipe)
    assert lowered.std() < x.std() < raised.std()


def test_local_strength_scales_stops_and_log_contrast():
    # Half strength is half a stop, not an arithmetic blend of sRGB pixels.
    image = torch.full((1, 3, 17, 23), 0.35)
    recipe = EditRecipe(exposure=torch.ones(1, 1, 4, 6))
    output = render(image, recipe, grade_strength=0, local_strength=0.5)
    expected = linear_to_srgb(srgb_to_linear(image) * (2**0.5))
    torch.testing.assert_close(output, expected, atol=3e-7, rtol=1e-5)
    # Scaling the map or the slider must implement the same contrast operation.
    image[..., ::2, ::2] += 0.025
    recipe = EditRecipe(log_contrast=torch.ones(1, 1, 4, 6))
    half_slider = render(image, recipe, local_strength=0.5)
    recipe.log_contrast *= 0.5
    torch.testing.assert_close(half_slider, render(image, recipe), atol=0, rtol=0)


@pytest.mark.parametrize("method", ["guided", "bilinear"])
def test_supervised_exposure_field_matches_actual_rendered_gain(method):
    torch.manual_seed(53)
    image = torch.rand(1, 3, 19, 31) * 0.3 + 0.15
    recipe = EditRecipe(
        exposure=torch.rand(1, 1, 5, 8) * 0.4 - 0.2,
        guide=torch.rand(1, 1, 5, 8),
        metadata={"upsampling": method},
    )
    actual_stops = torch.log2(
        luminance(srgb_to_linear(render(image, recipe))) / luminance(srgb_to_linear(image))
    )
    torch.testing.assert_close(exposure_field(image, recipe), actual_stops, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("order", ["global_first", "local_first"])
@pytest.mark.parametrize("upsampling", ["guided", "bilinear"])
def test_tiled_render_matches_whole_image_with_nonzero_maps(order, upsampling):
    torch.manual_seed(12)
    x = torch.rand(1, 3, 87, 131) * 0.8 + 0.05
    recipe = EditRecipe(
        luts=identity_lut(9)[None] * 0.95,
        exposure=torch.rand(1, 1, 13, 19) * 0.6 - 0.3,
        log_contrast=torch.rand(1, 1, 13, 19) * 0.6 - 0.3,
        guide=torch.rand(1, 1, 13, 19),
        metadata={"order": order, "upsampling": upsampling},
    )
    whole = render(x, recipe)
    tiled = render(x, recipe, tile_size=29)
    torch.testing.assert_close(tiled, whole, atol=7e-5, rtol=7e-5)


def test_gamut_aux_has_signal_after_output_clip():
    x = torch.full((1, 3, 13, 15), 0.95)
    recipe = EditRecipe(exposure=torch.ones(1, 1, 4, 4))
    output, aux = render_with_aux(x, recipe)
    assert output.max() == 1
    assert aux["pre_gamut"].max() > 1
    assert aux["gamut_penalty"] > 0


def test_omitting_unused_gamut_buffers_preserves_pixels_and_gradients():
    torch.manual_seed(63)
    image = (torch.rand(1, 3, 19, 27) * 0.8 + 0.1).requires_grad_()
    stops = torch.full((1, 1, 5, 7), 0.7, requires_grad=True)
    recipe = EditRecipe(stages=[EditRecipe(luts=identity_lut(5)[None] * 0.97), EditRecipe(exposure=stops)])
    with_aux, auxiliary = render_with_aux(image, recipe, tile_size=16)
    ordinary = render(image, recipe, tile_size=16)
    assert auxiliary["gamut_penalty"] > 0
    torch.testing.assert_close(ordinary, with_aux, atol=0, rtol=0)
    left = torch.autograd.grad(ordinary.square().mean(), (image, stops), retain_graph=True)
    right = torch.autograd.grad(with_aux.square().mean(), (image, stops))
    for actual, expected in zip(left, right, strict=False):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    _, omitted = render_with_aux(image, recipe, _compute_gamut=False)
    assert omitted["gamut_penalty"] == 0


def test_cascade_preserves_intermediate_highlights_until_final_clip():
    x = torch.full((1, 3, 11, 13), 0.9)
    brighter = EditRecipe(exposure=torch.ones(1, 1, 4, 4))
    darker = EditRecipe(exposure=-torch.ones(1, 1, 4, 4))
    recipe = EditRecipe(stages=[brighter, darker])
    torch.testing.assert_close(render(x, recipe), x, atol=2e-6, rtol=1e-5)
    lut_recipe = EditRecipe(stages=[brighter, EditRecipe(luts=identity_lut(9)[None]), darker])
    torch.testing.assert_close(render(x, lut_recipe), x, atol=3e-6, rtol=1e-5)


def test_recipe_roundtrip_and_detach():
    recipe = EditRecipe(
        exposure=torch.ones(1, 1, 4, 6, requires_grad=True), source_size=(24, 36), metadata={"version": 1}
    )
    parent = EditRecipe(stages=[recipe])
    restored = EditRecipe.from_state_dict(parent.detach().state_dict())
    assert restored.stages[0].source_size == (24, 36)
    assert not restored.stages[0].exposure.requires_grad


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_lut_and_local_renderer_mps_cpu_forward_backward_parity():
    torch.manual_seed(9)
    source = torch.rand(1, 3, 16, 19) * 0.6 + 0.2
    source_lut = identity_lut(5)[None] * 0.98
    results = []
    for device in ["cpu", "mps"]:
        x = source.to(device).detach().clone().requires_grad_()
        lut = source_lut.to(device).detach().clone().requires_grad_()
        exposure = torch.full((1, 1, 4, 5), 0.1, device=device, requires_grad=True)
        contrast = torch.full((1, 1, 4, 5), 0.1, device=device, requires_grad=True)
        recipe = EditRecipe(luts=lut, exposure=exposure, log_contrast=contrast)
        output = render(x, recipe)
        output.square().mean().backward()
        results.append(
            [output.detach().cpu(), x.grad.cpu(), lut.grad.cpu(), exposure.grad.cpu(), contrast.grad.cpu()]
        )
    for cpu, mps in zip(*results, strict=False):
        assert torch.isfinite(mps).all()
        torch.testing.assert_close(cpu, mps, atol=3e-5, rtol=3e-4)
