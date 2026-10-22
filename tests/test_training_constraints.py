import pytest
import torch

from photograder.color import srgb_to_linear
from photograder.renderer import apply_lut, identity_lut
from photograder.training_constraints import (
    additive_veil_corruption,
    lut_ramp_losses,
    lut_shape_losses,
)


@pytest.mark.parametrize("size", [17, 33])
def test_identity_and_affine_shape_constraints(size):
    lut = identity_lut(size, dtype=torch.float64)
    result = lut_shape_losses(lut)
    assert all(value.item() == 0 for value in result.values())
    affine = lut * 0.75 + 0.125
    assert all(value.item() == 0 for value in lut_shape_losses(affine).values())
    assert lut_shape_losses(1 - lut)["lut_monotonicity"].item() == pytest.approx(1 / 3)


def test_normalization_consistent_for_17_and_33_quadratic_tables():
    results = [lut_shape_losses(identity_lut(size, dtype=torch.float64).square()) for size in (17, 33)]
    for result in results:
        assert result["lut_monotonicity"].item() == 0
        assert result["lut_curvature"].item() == pytest.approx(4 / 3)
    unscaled = [
        lut_shape_losses(identity_lut(size, dtype=torch.float64).square(), normalize_derivatives=False)[
            "lut_curvature"
        ]
        for size in (17, 33)
    ]
    assert (unscaled[0] / unscaled[1]).item() == pytest.approx(16)


def test_shape_constraints_backward_and_two_cell_boundary():
    lut = (1 - identity_lut(17)).requires_grad_()
    sum(lut_shape_losses(lut).values()).backward()
    assert torch.isfinite(lut.grad).all() and lut.grad.abs().sum() > 0
    tiny = identity_lut(2).requires_grad_()
    curvature = lut_shape_losses(tiny)["lut_curvature"]
    assert curvature.item() == 0
    curvature.backward()
    assert torch.equal(tiny.grad, torch.zeros_like(tiny))


@pytest.mark.parametrize("size", [17, 33])
def test_dense_identity_rays_have_zero_loss_and_full_endpoints(size):
    seen = []

    def interpolate(image, luts):
        seen.append(image)
        return apply_lut(image, luts)

    result = lut_ramp_losses(identity_lut(size), interpolator=interpolate)
    assert all(value.item() == 0 for value in result.values())
    assert seen[0].shape == (1, 3, 7, 257)
    assert torch.equal(seen[0][0, :, 0, 0], torch.zeros(3))
    assert torch.equal(seen[0][0, :, 0, -1], torch.ones(3))
    assert seen[0].min() == 0 and seen[0].max() == 1


def test_ramps_detect_bright_reversal_and_preserve_gradients():
    identity = identity_lut(33, dtype=torch.float64)
    folded = identity.clone()
    # A single bright input slice has a large downward notch in every output.
    folded[:, 28, :, :] -= 0.2
    lut = folded.requires_grad_()
    losses = lut_ramp_losses(lut)
    assert losses["ramp_monotonicity"] > 0
    assert losses["ramp_curvature"] > 0
    sum(losses.values()).backward()
    assert torch.isfinite(lut.grad).all() and lut.grad.abs().sum() > 0


def test_neutral_ramp_known_reverse_slope_and_monotone_quadratic():
    identity = identity_lut(17, dtype=torch.float64)
    reverse = lut_ramp_losses(1 - identity, near_neutral_offset=0)
    assert reverse["ramp_monotonicity"].item() == pytest.approx(1)
    assert reverse["ramp_curvature"].item() == 0
    quadratic = lut_ramp_losses(identity.square(), near_neutral_offset=0)
    assert quadratic["ramp_monotonicity"].item() == 0
    assert quadratic["ramp_curvature"].item() > 0


def test_batched_lut_losses_match_per_table_average():
    a = identity_lut(17, dtype=torch.float64)
    b = 1 - a
    for function in (lut_shape_losses, lut_ramp_losses):
        together = function(torch.stack((a, b)))
        singles = [function(lut) for lut in (a, b)]
        for name in together:
            torch.testing.assert_close(together[name], (singles[0][name] + singles[1][name]) / 2)


def test_veil_linear_composition_support_and_identity_are_exact():
    image = torch.full((2, 3, 5, 7), 0.4, dtype=torch.float64)
    veil = torch.zeros_like(image)
    veil[:, :, 1:4, 2:5] = 0.08
    rng = torch.random.get_rng_state()
    result = additive_veil_corruption(image, veil)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(result["target"], image)
    assert result["support"].dtype == torch.bool
    assert result["support"].sum() == 18
    outside = ~result["support"].expand_as(image)
    assert torch.equal(result["image"][outside], image[outside])
    torch.testing.assert_close(srgb_to_linear(result["image"]), srgb_to_linear(image) + veil)
    assert not result["clipped"].any()
    assert torch.equal(result["available"], torch.ones_like(image))
    identity = additive_veil_corruption(image, torch.zeros(()))
    assert torch.equal(identity["image"], image)
    assert not identity["support"].any()


def test_veil_clipping_is_channelwise_and_does_not_erase_other_faults():
    image = torch.full((1, 3, 2, 3), 0.4, dtype=torch.float64)
    image[:, 0, 0, 0] = 0.99
    image[:, :, 1, 2] = 1  # Existing saturation where no veil is added.
    veil = torch.zeros_like(image)
    veil[..., 0, 0] = 0.1
    veil[..., 0, 1] = 0.1
    result = additive_veil_corruption(image, veil)
    assert result["clipped"].sum() == 1
    assert result["clipped"][0, 0, 0, 0]
    assert result["available"][0, 0, 0, 0] == 0
    assert result["available"][0, 1, 0, 0] == 1
    assert (result["image"][..., 0, 1] > image[..., 0, 1]).all()
    assert torch.equal(result["image"][..., 1, 2], image[..., 1, 2])
    assert torch.equal(result["available"][..., 1, 2], torch.ones(1, 3, dtype=image.dtype))


def test_veil_validity_mask_and_recoverable_gradient():
    image = torch.full((1, 3, 3, 4), 0.3, requires_grad=True)
    veil = torch.full((1, 3, 1, 1), 0.05, requires_grad=True)
    valid = torch.ones(1, 1, 3, 4)
    valid[..., 0, :] = 0
    valid[..., 1, :] = 0.5
    result = additive_veil_corruption(image, veil, valid_mask=valid)
    assert not result["support"][..., 0, :].any()
    assert torch.equal(result["image"][..., 0, :], image[..., 0, :])
    assert torch.equal(result["available"], valid.expand_as(image))
    ((result["image"] - result["target"]).abs() * result["available"]).mean().backward()
    assert torch.isfinite(image.grad).all()
    assert torch.isfinite(veil.grad).all() and veil.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [torch.zeros(3, 1, 1, 1), torch.zeros(3, 2, 3, 4), torch.zeros(2, 3, 4)])
def test_invalid_lut_shape_rejected(bad):
    for function in (lut_shape_losses, lut_ramp_losses):
        with pytest.raises(ValueError, match="LUT"):
            function(bad)


def test_invalid_veil_and_ramp_parameters_rejected():
    image = torch.zeros(1, 3, 2, 2)
    for veil in (torch.tensor(-0.1), torch.tensor(float("nan")), torch.zeros(4, 5)):
        with pytest.raises(ValueError, match="veil"):
            additive_veil_corruption(image, veil)
    with pytest.raises(ValueError, match="valid_mask"):
        additive_veil_corruption(image, torch.tensor(0.1), valid_mask=torch.zeros_like(image))
    with pytest.raises(ValueError, match="samples"):
        lut_ramp_losses(identity_lut(3), samples=2)
    with pytest.raises(ValueError, match="near_neutral_offset"):
        lut_ramp_losses(identity_lut(3), near_neutral_offset=0.5)
