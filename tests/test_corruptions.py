import pytest
import torch

from photograder.corruptions import FAMILIES, _srgb_to_linear, corrupt_batch


def photos():
    generator = torch.Generator().manual_seed(4)
    return torch.rand((4, 3, 32, 48), generator=generator) * 0.65 + 0.1


def test_deterministic_and_does_not_change_rng():
    image = photos()
    state = torch.random.get_rng_state()
    first = corrupt_batch(image, 3, allow_identity=False)
    second = corrupt_batch(image, 3, allow_identity=False)
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(state, torch.random.get_rng_state())
    assert not torch.equal(first["image"], image)


@pytest.mark.parametrize("family", FAMILIES)
def test_finite_support_no_new_clipping(family):
    image = photos()
    image[:, :, 0, 0] = 1
    image[:, :, 0, 1] = 0
    result = corrupt_batch(image, 18, families=(family,), allow_identity=False)
    outside = result["mask"].expand_as(image) == 0
    assert outside.any()
    assert torch.equal(image[outside], result["image"][outside])
    assert torch.isfinite(result["image"]).all()
    assert (result["image"] >= 0).all() and (result["image"] <= 1).all()
    assert not ((result["image"] >= 1) & (image < 0.999)).any()


def test_exact_exposure_inverse_and_padding():
    image = photos()
    valid = torch.ones((4, 1, 32, 48))
    valid[:, :, :8] = 0
    found = False
    for seed in range(10):
        result = corrupt_batch(image, seed, allow_identity=False, valid_mask=valid)
        assert not result["mask"][:, :, :8].any()
        for i, op in enumerate(result["operation"]):
            if op != "exposure":
                continue
            found = True
            recovered = _srgb_to_linear(result["image"][i]) * torch.exp2(result["known_inverse"][i])
            torch.testing.assert_close(recovered, _srgb_to_linear(image[i]), atol=2e-6, rtol=2e-6)
    assert found


def test_identity_is_exact_and_heldout_is_opt_in():
    image = photos()
    for seed in range(20):
        result = corrupt_batch(image, seed)
        for i, op in enumerate(result["operation"]):
            assert all(p["family"] != "edge_aware" for p in result["parameters"][i])
            if op == "identity":
                assert torch.equal(image[i], result["image"][i])
                assert not result["mask"][i].any()
                return
    pytest.fail("No identity sample generated")


def test_single_saturated_pixel_does_not_erase_contrast_fault():
    from photograder.corruptions import _apply_fault

    image = photos()[0]
    image[:, 16, 24] = 1
    mask = torch.ones_like(image[:1])
    result, delta, parameters = _apply_fault(image, mask, "contrast", torch.Generator().manual_seed(0))
    assert parameters["amplitude_scale"] > 0.5
    assert (result - image).abs().max() > 0.01
    assert not ((result >= 1) & (image < 0.999)).any()
