import torch

from photograder.color import linear_to_srgb, srgb_to_linear
from photograder.recipes import EditRecipe
from photograder.renderer import apply_lut, identity_lut, render_with_aux


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


def test_gamut_aux_has_signal_after_output_clip():
    x = torch.full((1, 3, 13, 15), 0.95)
    recipe = EditRecipe(luts=identity_lut(5)[None] * 1.5)
    output, aux = render_with_aux(x, recipe)
    assert output.max() == 1
    assert aux["pre_gamut"].max() > 1
    assert aux["gamut_penalty"] > 0
