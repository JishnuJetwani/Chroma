import itertools
import json

import pytest
import torch

from photograder.lut_safety import (
    apply_tetrahedral_lut,
    diagnose_neutral_ramp,
    smooth_lut_residual,
)
from photograder.renderer import identity_lut


def test_tetrahedral_identity_affine_broadcast_and_all_six_tetrahedra():
    # Each fractional ordering, plus values outside the input domain.
    pixels = list(itertools.permutations([0.17, 0.43, 0.79])) + [[-0.2, 0.3, 1.2], [1.3, -0.1, 0.6]]
    image = torch.tensor(pixels, dtype=torch.float64).T[None, :, None].repeat(2, 1, 1, 1)
    identity = identity_lut(2, dtype=torch.float64)
    matrix = torch.tensor([[0.8, -0.1, 0.2], [0.3, 0.5, -0.2], [-0.2, 0.1, 1.1]], dtype=torch.float64)
    bias = torch.tensor([0.02, 0.04, -0.1], dtype=torch.float64)
    lut = torch.einsum("ij,jabc->iabc", matrix, identity) + bias[:, None, None, None]
    for extrapolate in (False, True):
        source = image if extrapolate else image.clamp(0, 1)
        expected = torch.einsum("ij,bjhw->bihw", matrix, source) + bias[None, :, None, None]
        torch.testing.assert_close(apply_tetrahedral_lut(image, lut, extrapolate), expected)
        torch.testing.assert_close(apply_tetrahedral_lut(image, identity, extrapolate), source)
    batch = torch.stack((lut, lut + 0.1))
    output = apply_tetrahedral_lut(image, batch, extrapolate=True)
    torch.testing.assert_close(output[1] - output[0], torch.full_like(output[0], 0.1))


def test_tetrahedral_hits_every_grid_vertex_including_corners():
    torch.manual_seed(52)
    lut = torch.rand(2, 3, 5, 5, 5)
    coordinates = identity_lut(5).reshape(1, 3, 5, 25).expand(2, -1, -1, -1)
    output = apply_tetrahedral_lut(coordinates, lut)
    torch.testing.assert_close(output, lut.reshape(2, 3, 5, 25), atol=0, rtol=0)


def test_tetrahedral_gradients_for_input_and_lut():
    torch.manual_seed(61)
    image = (torch.rand(1, 3, 2, 3, dtype=torch.float64) * 0.8 + 0.1).requires_grad_()
    lut = (
        identity_lut(3, dtype=torch.float64)[None] + torch.rand(1, 3, 3, 3, 3, dtype=torch.float64) * 0.1
    ).requires_grad_()
    assert torch.autograd.gradcheck(apply_tetrahedral_lut, (image, lut), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("method", ["box", "gaussian"])
def test_residual_smoothing_preserves_identity_and_reduces_an_impulse(method):
    identity = identity_lut(5, dtype=torch.float64)
    assert torch.equal(smooth_lut_residual(identity, method=method, passes=2), identity)
    lut = identity.clone()
    lut[0, 2, 2, 2] += 0.2
    original = lut.clone()
    result = smooth_lut_residual(lut, method=method)
    residual = result - identity
    assert 0 < residual[0, 2, 2, 2] < 0.2
    assert residual[0, 2, 2, 1] > 0
    torch.testing.assert_close(residual.sum(), torch.tensor(0.2, dtype=torch.float64))
    assert torch.equal(lut, original)
    assert torch.equal(result[1:], identity[1:])
    batch = torch.stack((identity, lut)).requires_grad_()
    smooth_lut_residual(batch, method=method).square().sum().backward()
    assert torch.isfinite(batch.grad).all()


def test_diagnostic_catches_between_knot_reversal_and_does_not_mutate():
    lut = identity_lut(2)
    # All diagonal knots remain identity; off-diagonal corners cause a fold.
    lut[:, 0, 0, 1] = 2
    lut[:, 0, 1, 0] = 2
    lut[:, 1, 0, 0] = 2
    before = lut.clone()
    tri = diagnose_neutral_ramp(lut, samples=257)
    tetra = diagnose_neutral_ramp(lut, samples=257, interpolation="tetrahedral")
    assert tri["luts"][0]["negative_slope_fraction"] > 0
    assert tetra["luts"][0]["negative_slope_fraction"] == 0
    assert tetra["luts"][0]["min_slope_rgb"] == [1.0, 1.0, 1.0]
    json.dumps(tetra, allow_nan=False)
    image = torch.rand(2, 3, 3, 4)
    image_before = image.clone()
    apply_tetrahedral_lut(image, lut)
    smooth_lut_residual(lut)
    assert torch.equal(lut, before)
    assert torch.equal(image, image_before)


def test_invalid_shapes_and_options_raise_clear_errors():
    image, lut = torch.rand(2, 3, 2, 2), identity_lut(3)
    with pytest.raises(ValueError, match="batch"):
        apply_tetrahedral_lut(image, lut[None].repeat(3, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="D >= 2"):
        apply_tetrahedral_lut(image, torch.ones(3, 1, 1, 1))
    with pytest.raises(ValueError, match="dtype"):
        apply_tetrahedral_lut(image.double(), lut)
    with pytest.raises(ValueError, match="odd"):
        smooth_lut_residual(lut, kernel_size=2)
    with pytest.raises(ValueError, match="interpolation"):
        diagnose_neutral_ramp(lut, interpolation="nearest")
    with pytest.raises(ValueError, match="start"):
        diagnose_neutral_ramp(lut, start=0.8, end=0.2)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_tetrahedral_and_smoothing_cpu_mps_forward_backward_parity():
    torch.manual_seed(78)
    image = torch.rand(2, 3, 7, 9) * 1.4 - 0.2
    lut = identity_lut(5)[None] + torch.rand(1, 3, 5, 5, 5) * 0.05
    results = []
    for device in ("cpu", "mps"):
        x = image.to(device).detach().requires_grad_()
        table = lut.to(device).detach().requires_grad_()
        filtered = smooth_lut_residual(table)
        output = apply_tetrahedral_lut(x, filtered, extrapolate=True)
        output.square().mean().backward()
        results.append([output.detach().cpu(), x.grad.cpu(), table.grad.cpu()])
    for cpu, mps in zip(*results, strict=False):
        assert torch.isfinite(mps).all()
        torch.testing.assert_close(cpu, mps, atol=3e-5, rtol=3e-4)
