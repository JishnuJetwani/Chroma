"""RAW math invariants, with no private fixture or camera downloads."""

import numpy as np
import pytest

from photograder.raw_io import SRGB_TO_XYZ, RawDecodeError, camera_to_linear_srgb


def convert(image, **overrides):
    params = {
        "black_levels": [0, 0, 0],
        "white_levels": [1, 1, 1],
        "white_balance": [1, 1, 1],
        "rgb_xyz_matrix": np.linalg.inv(SRGB_TO_XYZ),
    }
    params.update(overrides)
    return camera_to_linear_srgb(np.asarray(image), **params)


def test_float_white_balance_preserves_highlights_and_negatives():
    result = convert([[[0.8, 0.5, -0.2]]], white_balance=[2, 1, 3])
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[[1.6, 0.5, -0.6]]], atol=1e-6)


def test_black_is_not_subtracted_twice_and_channel_ratios_are_preserved():
    black, white = [64, 128, 256], [1024, 2048, 4096]
    signal = np.array([100, 200, 300])
    result = convert([[signal]], black_levels=black, white_levels=white)
    np.testing.assert_allclose(result, [[signal / (4096 - 64)]], atol=1e-6)
    uncorrected = convert([[signal + black]], black_levels=black, white_levels=white, black_subtracted=False)
    np.testing.assert_allclose(uncorrected, result, atol=1e-6)


def test_common_scale_uses_fourth_green_black_and_explicit_libraw_white():
    result = convert([[[400, 400, 400]]], black_levels=[128, 128, 128, 127], white_levels=[4095] * 4)
    np.testing.assert_allclose(result, 400 / (4095 - 127), atol=1e-6)
    result = convert(
        [[[400, 400, 400]]],
        black_levels=[128] * 4,
        white_levels=[4095, 4094, 4093, 4092],
        normalization_scale=4000,
    )
    np.testing.assert_allclose(result, 0.1, atol=1e-6)


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan"), [1, 1, 1]])
def test_invalid_common_scale_is_rejected(scale):
    with pytest.raises(RawDecodeError):
        convert([[[0.2, 0.2, 0.2]]], normalization_scale=scale)


def test_camera_matrix_normalization_preserves_balanced_neutral():
    camera_xyz = [[0.6602, -0.0841, -0.0939], [-0.4472, 1.2458, 0.2247], [-0.0975, 0.2039, 0.6148], [0, 0, 0]]
    wb = np.array([2.0, 1.0, 1.5])
    result = convert([[np.ones(3) * 0.4 / wb]], white_balance=wb, rgb_xyz_matrix=camera_xyz)
    np.testing.assert_allclose(result, 0.4, atol=1e-6)
    # The same matrix's red response extends outside display range.
    response = convert([[[1.0, 0.0, 0.0]]], rgb_xyz_matrix=camera_xyz)
    assert response[0, 0, 0] > 1 and response[0, 0, 1] < 0


@pytest.mark.parametrize(
    "matrix", [np.zeros((4, 3)), np.ones((3, 3)), np.full((3, 3), np.nan), np.ones((4, 4))]
)
def test_invalid_matrix_is_rejected(matrix):
    with pytest.raises(RawDecodeError):
        convert([[[0.2, 0.2, 0.2]]], rgb_xyz_matrix=matrix)


@pytest.mark.parametrize(
    "override",
    [
        {"white_balance": [0, 1, 1]},
        {"white_balance": [1, float("nan"), 1]},
        {"white_levels": [0, 1, 1]},
        {"black_levels": [-1, 0, 0]},
    ],
)
def test_invalid_levels_and_wb_are_rejected(override):
    with pytest.raises(RawDecodeError):
        convert([[[0.2, 0.2, 0.2]]], **override)
