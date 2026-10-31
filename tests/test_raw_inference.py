import hashlib
from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image

from photograder import raw_inference
from photograder.inference import tensor_image
from photograder.raw_inference import RawGrader, is_raw_checkpoint
from photograder.raw_io import RAW_PIPELINE_VERSION, RawImage
from photograder.raw_model import RawEditRecipe, RawGradingModel, RawModelConfig, baseline_raw, render_raw


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def checkpoint(tmp_path):
    network = RawGradingModel(RawModelConfig(preview_size=64, map_size=16))
    path = tmp_path / "raw-trained.pt"
    torch.save(
        {
            "model": network.state_dict(),
            "model_config": asdict(network.config),
            "raw_pipeline_version": RAW_PIPELINE_VERSION,
            "metadata": {"trained": True, "pilot": True},
        },
        path,
    )
    return path


@pytest.fixture
def scene(tmp_path, monkeypatch):
    path = tmp_path / "original.RAF"
    path.write_bytes(b"Test fixture; decoder is mocked, not a real RAW file.")
    pixels = np.random.default_rng(15).uniform(-0.08, 4, (81, 137, 3)).astype(np.float32)
    calls = []

    def decode(source, **kwargs):
        calls.append((source, kwargs))
        return RawImage(
            pixels,
            {
                "pipeline_version": RAW_PIPELINE_VERSION,
                "working_space": "scene-linear-sRGB",
                "sensor_clip_fraction": 0.002,
            },
        )

    monkeypatch.setattr(raw_inference, "decode_raw", decode)
    return path, pixels, calls


def test_checkpoint_rejects_untrained_and_mismatched_pipeline(checkpoint, tmp_path):
    assert is_raw_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    payload["metadata"]["trained"] = False
    untrained = tmp_path / "untrained.pt"
    torch.save(payload, untrained)
    with pytest.raises(ValueError, match="not been trained"):
        RawGrader(untrained, "cpu")
    payload["metadata"]["trained"] = True
    payload["raw_pipeline_version"] = "different-normalization"
    mismatch = tmp_path / "mismatch.pt"
    torch.save(payload, mismatch)
    with pytest.raises(ValueError, match="different decoding pipeline"):
        RawGrader(mismatch, "cpu")


def test_analysis_preserves_cpu_float_native_and_uses_direct_antialiased_preview(
    checkpoint, scene, monkeypatch
):
    source, pixels, calls = scene
    grader = RawGrader(checkpoint, "cpu", preview_size=96, tile_size=23)
    observed = []
    analyze = grader.model.analyze

    def capture(image, *args, **kwargs):
        observed.append(image.clone())
        return analyze(image, *args, **kwargs)

    monkeypatch.setattr(grader.model, "analyze", capture)
    analysis = grader.analyze(source)
    assert len(calls) == 1 and calls[0][1]["max_pixels"] == 60_000_000
    assert analysis.linear_rgb.device.type == "cpu"
    assert analysis.linear_rgb.dtype == torch.float32
    assert analysis.linear_rgb.min() < 0 and analysis.linear_rgb.max() > 1
    np.testing.assert_array_equal(analysis.linear_rgb[0].permute(1, 2, 0).numpy(), pixels)
    assert analysis.source_size == (81, 137)
    assert analysis.preview.size == (96, 57)
    expected = torch.nn.functional.interpolate(
        analysis.linear_rgb, size=(38, 64), mode="bilinear", align_corners=False, antialias=True
    )
    torch.testing.assert_close(observed[0], expected, atol=0, rtol=0)
    np.testing.assert_array_equal(
        np.asarray(analysis.preview), np.asarray(tensor_image(baseline_raw(analysis.preview_linear)))
    )


def test_cached_sliders_export_and_recipe_never_reanalyze_or_decode(checkpoint, scene, tmp_path, monkeypatch):
    source, _, calls = scene
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    grader = RawGrader(checkpoint, "cpu", preview_size=96, tile_size=23)
    analysis = grader.analyze(source)
    analysis.recipe.exposure_ev = torch.full_like(analysis.recipe.exposure_ev, -1)

    def forbidden(*args, **kwargs):
        raise AssertionError("A cached slider or export must not decode or analyze again")

    monkeypatch.setattr(grader.model, "analyze", forbidden)
    monkeypatch.setattr(raw_inference, "decode_raw", forbidden)
    full = np.asarray(grader.render_preview(analysis, 1))
    neutral = np.asarray(grader.render_preview(analysis, 0))
    assert full.mean() < neutral.mean()
    assert len(calls) == 1
    output = grader.export(analysis, tmp_path / "result.png", 0.7, 0.25)
    with Image.open(output["path"]) as image:
        assert image.size == (137, 81)
        assert image.info.get("icc_profile")
        expected = tensor_image(
            render_raw(analysis.linear_rgb, analysis.recipe, strength=0.7, exposure_bias_ev=0.25)
        )
        np.testing.assert_allclose(np.asarray(image), np.asarray(expected), atol=1, rtol=0)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    recipe_path = tmp_path / "recipe.pt"
    grader.save_recipe(analysis, recipe_path)
    payload = torch.load(recipe_path, weights_only=True)
    restored = RawEditRecipe.from_state_dict(payload["recipe"])
    torch.testing.assert_close(restored.exposure_ev, analysis.recipe.exposure_ev)
    assert payload["raw_pipeline_version"] == RAW_PIPELINE_VERSION
    assert payload["source_sha256"] == before
    with pytest.raises(ValueError, match="original is preserved"):
        grader.export(analysis, source)
    with pytest.raises(FileExistsError):
        grader.export(analysis, output["path"])


def test_native_export_streams_global_coordinate_tiles(checkpoint, scene, tmp_path, monkeypatch):
    source, _, _ = scene
    grader = RawGrader(checkpoint, "cpu", preview_size=64, tile_size=23)
    analysis = grader.analyze(source)
    # Explicit smooth ramp reveals map coordinates restarting at tile edges.
    analysis.linear_rgb = torch.full_like(analysis.linear_rgb, 0.4)
    analysis.recipe.local_exposure_ev = torch.linspace(-1, 1, 16)[None, None, None].expand(1, 1, 10, 16)
    shapes = []
    real_renderer = raw_inference.render_raw_tile

    def capture(image, *args, **kwargs):
        shapes.append((image.shape[-2:], kwargs["origin"], kwargs["full_size"]))
        return real_renderer(image, *args, **kwargs)

    monkeypatch.setattr(raw_inference, "render_raw_tile", capture)
    path = grader.export(analysis, tmp_path / "tiles.png")["path"]
    assert len(shapes) > 4 and all(max(size) <= 23 for size, _, _ in shapes)
    assert shapes[-1][1] == (69, 115)
    assert all(full_size == (81, 137) for _, _, full_size in shapes)
    with Image.open(path) as image:
        actual = np.asarray(image)
    expected = np.asarray(tensor_image(render_raw(analysis.linear_rgb, analysis.recipe)))
    np.testing.assert_allclose(actual, expected, atol=1, rtol=0)
    assert np.diff(actual.astype(float).mean(axis=(0, 2))).min() >= 0


def test_tiff_export_is_16_bit_rgb_srgb(checkpoint, scene, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    source, _, _ = scene
    grader = RawGrader(checkpoint, "cpu", preview_size=64, tile_size=31)
    analysis = grader.analyze(source)
    result = grader.export(analysis, tmp_path / "developed.tif", 0)
    assert result["bit_depth"] == 16
    with tifffile.TiffFile(result["path"]) as file:
        pixels = file.asarray()
        assert pixels.dtype == np.uint16 and pixels.shape == (81, 137, 3)
        assert file.pages[0].tags[34675].value
    expected = baseline_raw(analysis.linear_rgb)[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(pixels / 65535.0, expected, atol=1 / 65535, rtol=0)
    assert np.any(pixels % 257 != 0), "16-bit output must not be expanded from an 8-bit image"


def test_app_opens_raw_first_and_retains_legacy_tab(checkpoint, tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from photograder import app
    from photograder.models import ModelConfig, build_model
    from photograder.runtime import save_checkpoint

    legacy_path = tmp_path / "jpeg.pt"
    save_checkpoint(
        legacy_path, build_model(ModelConfig(architecture="fixed", lut_size=5)), metadata={"trained": True}
    )
    monkeypatch.setattr(app, "JPEG_CHECKPOINT_CANDIDATES", (legacy_path,))
    demo = app.create_app(checkpoint, device="cpu", export_dir=tmp_path / "exports")
    components = demo.config["components"]
    assert any(item["type"] == "tabs" and item["props"]["selected"] == "raw" for item in components)
    labels = [item["props"].get("label") for item in components]
    assert "RAW" in labels and "JPEG / PNG" in labels
    assert "Camera RAW photo" in labels and "Your photo" in labels
    assert "Exposure adjustment (stops)" in labels
    demo.close()


def test_dng_inference_applies_exact_training_crop_and_orientation(checkpoint, tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "camera.dng"
    with tifffile.TiffWriter(path) as writer:
        writer.write(
            np.zeros((2, 3, 3), dtype=np.uint8),
            photometric="rgb",
            metadata=None,
            subifds=1,
            extratags=[(50706, "B", 4, (1, 4, 0, 0), False), (274, "H", 1, 6, False)],
        )
        writer.write(
            np.zeros((10, 12), dtype=np.uint16),
            photometric=32803,
            metadata=None,
            extratags=[
                (50829, "I", 4, (1, 2, 10, 12), False),
                (50719, "2I", 2, (1, 1, 1, 1), False),
                (50720, "2I", 2, (6, 1, 4, 1), False),
            ],
        )
    raw = np.arange(9 * 10 * 3, dtype=np.float32).reshape(9, 10, 3) / 64 - 0.1
    calls = []
    metadata = {
        "pipeline_version": RAW_PIPELINE_VERSION,
        "orientation_applied": False,
        "original_geometry": {
            "width": 10,
            "height": 9,
            "left_margin": 2,
            "top_margin": 1,
            "raw_width": 12,
            "raw_height": 10,
        },
    }

    def decode(source, **kwargs):
        calls.append(kwargs)
        return RawImage(raw, metadata)

    monkeypatch.setattr(raw_inference, "decode_raw", decode)
    grader = RawGrader(checkpoint, "cpu", preview_size=64)
    analysis = grader.analyze(path)
    assert calls == [{"max_pixels": 60_000_000, "apply_orientation": False}]
    expected = np.rot90(raw[1:5, 1:7], -1)
    np.testing.assert_array_equal(analysis.linear_rgb[0].permute(1, 2, 0).numpy(), expected)
    assert analysis.source_size == (6, 4)
    audit = analysis.raw_metadata["development_geometry"]
    assert audit["raw_visible_crop_xyxy"] == [1, 1, 7, 5]
    assert audit["orientation_applied"] == 6
    result = grader.export(analysis, tmp_path / "crop.png", 0)
    with Image.open(result["path"]) as image:
        assert image.size == (4, 6)
