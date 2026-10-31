"""Precision and geometry checks using tiny synthetic TIFF/DNG fixtures."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import ImageCms

from photograder.raw_acquisition import dng_crop_metadata
from photograder.raw_data import (
    RENDER_VERSION,
    aligned_pair,
    icc_rgb16_to_srgb_float,
    load_expert_tiff,
    orient_image,
    prepare_raw_pair,
)
from photograder.raw_io import RawImage


@pytest.fixture
def srgb_profile():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def write_target(path, pixels, profile, orientation=1):
    tags = [(274, "H", 1, orientation, False)]
    if profile:
        tags.append((34675, "B", len(profile), profile, False))
    tifffile.imwrite(path, pixels, photometric="rgb", metadata=None, extratags=tags)


def write_dng_tags(path, *, origin=(1, 1), orientation=6):
    # Small synthetic CFA data exercises real TIFF SubIFD metadata; it is not a
    # photographic RAW fixture and is never passed to LibRaw for demosaicing.
    with tifffile.TiffWriter(path) as writer:
        writer.write(
            np.zeros((2, 3, 3), dtype=np.uint8),
            photometric="rgb",
            metadata=None,
            subifds=1,
            extratags=[(50706, "B", 4, (1, 4, 0, 0), False), (274, "H", 1, orientation, False)],
        )
        writer.write(
            np.zeros((10, 12), dtype=np.uint16),
            photometric=32803,
            metadata=None,
            extratags=[
                (50829, "I", 4, (1, 2, 10, 12), False),
                (50719, "2I", 2, (*origin, 1, 1), False),
                (50720, "2I", 2, (6, 1, 4, 1), False),
            ],
        )


def decoder_metadata(*, width=10, height=9, left=2, top=1, sensor_width=12, sensor_height=10):
    return {
        "orientation_applied": False,
        "original_geometry": {
            "width": width,
            "height": height,
            "left_margin": left,
            "top_margin": top,
            "raw_width": sensor_width,
            "raw_height": sensor_height,
        },
    }


def crop_metadata(*, orientation=6):
    return {
        "fullsensor_size_wh": [12, 10],
        "active_area_yxyx": [1, 2, 10, 12],
        "default_crop_origin_xy": [1, 1],
        "default_crop_size_wh": [6, 4],
        "default_scale_xy": [1, 1],
        "exif_orientation": orientation,
    }


def test_icc_transform_uses_low_sixteen_bit_samples(srgb_profile):
    # Every pixel has the same high eight bits. An 8-bit detour would collapse
    # the entire row, although all 256 low-byte distinctions exist in the TIFF.
    samples = np.arange(256, dtype=np.uint16) + 32768
    pixels = np.repeat(samples[None, :, None], 3, axis=2)
    result, metadata = icc_rgb16_to_srgb_float(pixels, srgb_profile)
    assert result.dtype == np.float32
    assert len(np.unique(result[0, :, 0])) > 240
    np.testing.assert_allclose(result, pixels.astype(np.float32) / 65535, atol=3e-5)
    assert metadata["source_bits"] == 16 and metadata["target_storage"] == "float32"
    assert metadata["icc_sha256"] == hashlib.sha256(srgb_profile).hexdigest()


def test_target_tiff_keeps_precision_and_applies_its_orientation(tmp_path, srgb_profile):
    pixels = np.arange(18, dtype=np.uint16).reshape(2, 3, 3) * 1000 + 16384
    path = tmp_path / "expert.tif"
    write_target(path, pixels, srgb_profile, orientation=6)
    result, metadata = load_expert_tiff(path)
    expected, _ = icc_rgb16_to_srgb_float(pixels, srgb_profile)
    np.testing.assert_allclose(result, np.rot90(expected, -1))
    assert result.shape == (3, 2, 3)
    assert metadata["stored_dimensions"] == [3, 2] and metadata["orientation"] == 6


@pytest.mark.parametrize("dtype,profile", [(np.uint8, True), (np.uint16, False)])
def test_expert_rejects_quantized_or_unprofiled_targets(tmp_path, srgb_profile, dtype, profile):
    path = tmp_path / "bad-expert.tif"
    write_target(path, np.zeros((2, 3, 3), dtype=dtype), srgb_profile if profile else None)
    with pytest.raises(ValueError, match="16-bit|ICC profile"):
        load_expert_tiff(path)


def test_expert_rejects_invalid_icc(srgb_profile):
    with pytest.raises(ValueError, match="ICC profile"):
        icc_rgb16_to_srgb_float(np.zeros((2, 3, 3), dtype=np.uint16), b"not an ICC profile")


def test_dng_crop_reads_cfa_subifd_and_rational_tags(tmp_path):
    path = tmp_path / "metadata-fixture.dng"
    write_dng_tags(path)
    metadata = dng_crop_metadata(path)
    assert metadata["fullsensor_size_wh"] == [12, 10]
    assert metadata["ifd_index_path"] == [0, 0]
    assert metadata["default_crop_origin_xy"] == [1, 1]
    assert metadata["default_crop_size_wh"] == [6, 4]
    assert metadata["active_area_yxyx"] == [1, 2, 10, 12]
    assert metadata["exif_orientation"] == 6
    assert metadata["ifd_offset"] > 8  # It selected the sensor, not IFD0's RGB preview.


def test_fractional_dng_crop_is_not_rounded(tmp_path):
    path = tmp_path / "fractional-fixture.dng"
    write_dng_tags(path, origin=(1, 2))
    with pytest.raises(ValueError, match="Noninteger"):
        dng_crop_metadata(path)
    assert dng_crop_metadata(path, require_integer_crop=False)["default_crop_origin_xy"] == [0.5, 1]


def test_exact_crop_avoids_one_pixel_centering_error():
    raw = np.repeat(np.arange(12, dtype=np.float32)[None, :, None], 10, axis=0)
    raw = np.repeat(raw, 3, axis=2)
    crop = {
        "fullsensor_size_wh": [12, 10],
        "active_area_yxyx": [0, 0, 10, 12],
        "default_crop_origin_xy": [2, 1],
        "default_crop_size_wh": [6, 8],
        "default_scale_xy": [1, 1],
        "exif_orientation": 1,
    }
    result, _, audit = aligned_pair(
        raw,
        np.zeros((8, 6, 3), dtype=np.float32),
        dng_geometry=crop,
        decoder_metadata=decoder_metadata(width=12, height=10, left=0, top=0),
    )
    np.testing.assert_array_equal(result[0, :, 0], [2, 3, 4, 5, 6, 7])
    assert audit["raw_visible_crop_xyxy"] == [2, 1, 8, 9]
    assert "raw_center_crop" not in audit


def test_crop_accounts_for_active_area_and_libraw_margins_before_rotation():
    raw = np.arange(9 * 10 * 3, dtype=np.float32).reshape(9, 10, 3)
    result, _, audit = aligned_pair(
        raw,
        np.zeros((6, 4, 3), dtype=np.float32),
        dng_geometry=crop_metadata(),
        decoder_metadata=decoder_metadata(),
    )
    np.testing.assert_array_equal(result, np.rot90(raw[1:5, 1:7], -1))
    assert audit["raw_sensor_crop_xyxy"] == [3, 2, 9, 6]
    assert audit["raw_visible_crop_xyxy"] == [1, 1, 7, 5]
    assert audit["orientation_applied"] == 6


@pytest.mark.parametrize(
    "orientation,expected",
    [
        (1, [[0, 1, 2], [3, 4, 5]]),
        (2, [[2, 1, 0], [5, 4, 3]]),
        (3, [[5, 4, 3], [2, 1, 0]]),
        (4, [[3, 4, 5], [0, 1, 2]]),
        (5, [[0, 3], [1, 4], [2, 5]]),
        (6, [[3, 0], [4, 1], [5, 2]]),
        (7, [[5, 2], [4, 1], [3, 0]]),
        (8, [[2, 5], [1, 4], [0, 3]]),
    ],
)
def test_all_exif_orientations(orientation, expected):
    image = np.repeat(np.arange(6).reshape(2, 3, 1), 3, axis=2)
    np.testing.assert_array_equal(orient_image(image, orientation)[:, :, 0], expected)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"default_crop_origin_xy": [0.5, 1]}, "Noninteger"),
        ({"default_scale_xy": [2, 1]}, "DefaultScale"),
        ({"default_crop_size_wh": None}, "Incomplete"),
        ({"default_crop_origin_xy": [7, 1]}, "outside"),
        ({"fullsensor_size_wh": [13, 10]}, "sensor dimensions differ"),
    ],
)
def test_invalid_geometry_never_uses_center_fallback(mutation, error):
    crop = {**crop_metadata(), **mutation}
    with pytest.raises(ValueError, match=error):
        aligned_pair(
            np.zeros((9, 10, 3)), np.zeros((6, 4, 3)), dng_geometry=crop, decoder_metadata=decoder_metadata()
        )


def test_exact_crop_rejects_target_size_mismatch():
    with pytest.raises(ValueError, match="Exact DNG crop geometry"):
        aligned_pair(
            np.zeros((9, 10, 3)),
            np.zeros((6, 5, 3)),
            dng_geometry=crop_metadata(),
            decoder_metadata=decoder_metadata(),
        )


def test_decoder_must_be_unoriented_and_match_visible_dimensions():
    metadata = decoder_metadata()
    metadata["orientation_applied"] = True
    with pytest.raises(ValueError, match="unoriented"):
        aligned_pair(
            np.zeros((9, 10, 3)), np.zeros((6, 4, 3)), dng_geometry=crop_metadata(), decoder_metadata=metadata
        )
    with pytest.raises(ValueError, match="visible sensor rectangle"):
        aligned_pair(
            np.zeros((10, 9, 3)),
            np.zeros((6, 4, 3)),
            dng_geometry=crop_metadata(),
            decoder_metadata=decoder_metadata(),
        )


def test_missing_crop_fallback_is_oriented_bounded_and_recorded():
    crop = {**crop_metadata(), "default_crop_origin_xy": None, "default_crop_size_wh": None}
    raw = np.arange(9 * 10 * 3).reshape(9, 10, 3)
    result, _, audit = aligned_pair(
        raw, np.zeros((8, 7, 3)), dng_geometry=crop, decoder_metadata=decoder_metadata()
    )
    np.testing.assert_array_equal(result, np.rot90(raw, -1)[1:9, 1:8])
    assert "fallback" in audit["crop_method"]
    with pytest.raises(ValueError, match="Unverified pair geometry"):
        aligned_pair(
            raw, np.zeros((8, 7, 3)), max_border=1, dng_geometry=crop, decoder_metadata=decoder_metadata()
        )


def test_prepare_preserves_exact_crop_audit_and_float_headroom(tmp_path, monkeypatch, srgb_profile):
    source, target, output = (tmp_path / name for name in ("fixture.dng", "expert.tif", "sample.npz"))
    write_dng_tags(source)
    write_target(target, np.full((6, 4, 3), 32768, dtype=np.uint16), srgb_profile)
    linear = np.broadcast_to(np.array([-0.25, 1.5, 0.3], dtype=np.float32), (9, 10, 3)).copy()
    calls = []

    def decode(path, **kwargs):
        calls.append((path, kwargs))
        return RawImage(linear, copy.deepcopy(decoder_metadata()))

    monkeypatch.setattr("photograder.raw_io.decode_raw", decode)
    result = prepare_raw_pair(source, target, output, scene_id="fixture:one", split="train", size=6)
    assert calls == [(source, {"apply_orientation": False})]
    assert result["render_version"] == RENDER_VERSION
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with np.load(output, allow_pickle=False) as cache:
        assert cache["image"].dtype == np.float32 and cache["target"].dtype == np.float32
        assert cache["image"].min() < 0 and cache["image"].max() > 1
        assert cache["valid_mask"].sum() == 24
        metadata = json.loads(str(cache["metadata"]))
    assert metadata["geometry"]["raw_visible_crop_xyxy"] == [1, 1, 7, 5]
    assert metadata["geometry"]["dng_crop_metadata"]["ifd_index_path"] == [0, 0]
    assert metadata["geometry"]["orientation_applied"] == 6
    assert metadata["content_box"] == [1, 0, 5, 6]


def test_available_official_header_metadata_matches_target_dimensions():
    # Optional integration evidence is data-only and can be absent on CI. No
    # network or original image download is performed by this test.
    evidence = Path(__file__).resolve().parents[1] / "data/raw_fivek_plan/dng-crop-probe.json"
    if not evidence.is_file():
        pytest.skip("Official bounded-header probe metadata is not present")
    expected = {"a0001": [3008, 2000], "a0005": [2336, 3504], "a0007": [2848, 4272]}
    for row in json.loads(evidence.read_text()):
        photo = Path(row["url"]).name[:5]
        crop = row["crop"]
        assert crop["ifd_index_path"] == [0, 0]
        size = crop["default_crop_size_wh"]
        oriented = size[::-1] if crop["exif_orientation"] in (5, 6, 7, 8) else size
        assert oriented == expected[photo]
