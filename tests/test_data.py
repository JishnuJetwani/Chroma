import copy
import json

import numpy as np
import pytest
from PIL import Image

from photograder.data import PairedDataset, render_s24_linear, sha256_file, validate_manifest, write_manifest


def example(tmp_path, scene="fixture:one", split="train", style=0, size=(40, 20)):
    # This patterned fixture tests data handling; it is not research photo data.
    array = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[1], size[0], 3)
    left = tmp_path / f"{scene.replace(':', '_')}_input.png"
    right = tmp_path / f"{scene.replace(':', '_')}_target.png"
    Image.fromarray(array).save(left)
    Image.fromarray(255 - array).save(right)
    return dict(
        source_id="test-fixture",
        scene_id=scene,
        input=left.name,
        target=right.name,
        style_id=style,
        style_name=f"Style {style}",
        split=split,
        render_version="test-v1",
        sha256={"input": sha256_file(left), "target": sha256_file(right)},
    )


def test_padding_and_validity(tmp_path):
    row = example(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([row], manifest)
    sample = PairedDataset(manifest, "train", image_size=32)[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["valid_mask"].sum() == 32 * 16
    assert sample["content_box"].tolist() == [0, 8, 32, 16]
    assert sample["original_size"].tolist() == [40, 20]
    assert validate_manifest(manifest)["rows"] == 1


def test_scene_style_leakage_and_immutable_split(tmp_path):
    row = example(tmp_path)
    bad = copy.deepcopy(row)
    bad.update(split="test", style_id=1, style_name="Other")
    manifest = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError, match="Scene leakage"):
        write_manifest([row, bad], manifest)
    write_manifest([row], manifest)
    with pytest.raises(ValueError, match="immutable"):
        write_manifest([bad], manifest)


def test_checksum_and_geometry(tmp_path):
    row = example(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([row], manifest)
    Image.new("RGB", (41, 20)).save(tmp_path / row["target"])
    with pytest.raises(ValueError, match="Checksum"):
        validate_manifest(manifest)
    with pytest.raises(ValueError, match="dimensions"):
        PairedDataset(manifest, "train", 32)[0]


def test_exif_orientation_and_privacy_mask(tmp_path):
    row = example(tmp_path)
    mask = np.ones((20, 40), dtype=np.uint8) * 255
    mask[:, :20] = 0
    Image.fromarray(mask).save(tmp_path / "valid.png")
    row["valid_mask"] = "valid.png"
    row["sha256"]["valid_mask"] = sha256_file(tmp_path / "valid.png")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([row], manifest)
    assert PairedDataset(manifest, "train", 40)[0]["valid_mask"].sum() == 400


def test_s24_uses_camera_wb_and_linear_transfer():
    raw = np.ones((2, 3, 3), dtype=np.float32) * 0.1
    metadata = {"cam_illum": [0.5, 1, 2], "ccm": np.eye(3).tolist(), "gt_illum": [1, 1, 1]}
    srgb, clipped = render_s24_linear(raw, metadata)
    expected = 1.055 * np.array([0.2, 0.1, 0.05]) ** (1 / 2.4) - 0.055
    np.testing.assert_allclose(srgb[0, 0], expected, atol=1e-6)
    assert clipped == 0
    with pytest.raises(ValueError, match="16-bit"):
        render_s24_linear(np.ones((2, 3, 3), dtype=np.uint8), metadata)


def test_exif_rotation_is_applied_before_geometry(tmp_path):
    row = example(tmp_path)
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (20, 40), (80, 120, 200)).save(tmp_path / row["input"], exif=exif)
    row["sha256"]["input"] = sha256_file(tmp_path / row["input"])
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([row], manifest)
    assert validate_manifest(manifest)["rows"] == 1
    assert PairedDataset(manifest, "train", 40)[0]["original_size"].tolist() == [40, 20]


def test_invalid_icc_profile_is_not_silently_ignored(tmp_path):
    from photograder.data import load_srgb

    path = tmp_path / "broken.png"
    Image.new("RGB", (16, 16)).save(path, icc_profile=b"not an ICC profile")
    with pytest.raises(OSError, match="profile"):
        load_srgb(path)


def test_remote_zip_guards_and_per_member_integrity(tmp_path):
    import io
    import zipfile

    from photograder.acquisition import RemoteZipReader, _member_bytes, _safe_member, _store_member

    contents = io.BytesIO()
    with zipfile.ZipFile(contents, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("input/photo.jpg", b"sample fixture data")
        archive.writestr("../unsafe.txt", b"bad")
    payload = contents.getvalue()

    class MemoryReader(RemoteZipReader):
        def range(self, start, size):
            return payload[start : start + size]

    reader = MemoryReader("memory://fixture", len(payload))
    archive = zipfile.ZipFile(reader)
    info = archive.getinfo("input/photo.jpg")
    assert _member_bytes(reader, info) == b"sample fixture data"
    destination = tmp_path / "photo.jpg"
    assert _store_member(reader, info, destination)
    assert not _store_member(reader, info, destination)
    destination.write_bytes(b"corrupt resume")
    with pytest.raises(ValueError, match="corrupt"):
        _store_member(reader, info, destination)
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_member(archive.getinfo("../unsafe.txt"))
    bounded = RemoteZipReader("invalid://never-called", 100)
    with pytest.raises(ValueError, match="invalid"):
        bounded.range(90, 11)


def test_remote_zip_refuses_full_download_response(monkeypatch):
    from photograder.acquisition import RemoteZipReader

    reader = RemoteZipReader("https://example.test/fixture.zip", 100)

    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, *args):
            pytest.fail("Must never consume ignored-range full archive")

    monkeypatch.setattr(reader.session, "get", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="ignored Range"):
        reader.range(90, 10)


@pytest.mark.parametrize("styles", [(0,), (2, 5)])
def test_s24_import_preserves_precision_splits_and_privacy(tmp_path, styles, monkeypatch):
    png = pytest.importorskip("png")
    # Tiny fixture tests must not depend on the host having a dataset-sized
    # reserve. Separate StorageGuard tests exercise real admission decisions.
    import shutil

    usage_type = type(shutil.disk_usage(tmp_path))
    monkeypatch.setattr(
        "photograder.acquisition.shutil.disk_usage", lambda path: usage_type(100 * 2**30, 0, 100 * 2**30)
    )
    from photograder.data import _read_rgb16, import_s24, read_manifest

    source, output = tmp_path / "official_fixture", tmp_path / "prepared"
    for index, split in enumerate(("train", "val", "test")):
        for directory in (
            "denoised_raw_images",
            "data",
            "blur_masks",
            *(f"srgb_images_style_{style}" for style in styles),
        ):
            (source / split / directory).mkdir(parents=True)
        identity = f"{split}_fixture"
        raw = np.arange(16 * 24 * 3, dtype=np.uint16).reshape(16, 24, 3) + index * 1234
        raw_path = source / split / "denoised_raw_images" / (identity + ".png")
        with raw_path.open("wb") as stream:
            png.Writer(24, 16, bitdepth=16, greyscale=False).write(stream, raw.reshape(16, -1))
        np.testing.assert_allclose(_read_rgb16(raw_path), raw / 65535, atol=1e-7)
        metadata = {"cam_illum": [1, 1, 1], "ccm": np.eye(3).tolist(), "orientation": 1}
        (source / split / "data" / (identity + ".json")).write_text(json.dumps(metadata))
        rendered, _ = render_s24_linear(raw.astype(np.float32) / 65535, metadata)
        for style in styles:
            Image.fromarray((rendered * 255).astype(np.uint8)).save(
                source / split / f"srgb_images_style_{style}" / (identity + ".jpg")
            )
        blur = np.zeros((16, 24), dtype=np.uint8)
        blur[:, :6] = 1  # Official masks may use literal1 rather than255.
        Image.fromarray(blur).save(source / split / "blur_masks" / (identity + ".png"))
    manifest = import_s24(source, output, styles=styles)
    rows = read_manifest(manifest)
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert {r["style_name"] for r in rows} == {f"S24 Style {style}" for style in styles}
    assert {r["style_id"]: r["source_style_id"] for r in rows} == dict(enumerate(styles))
    assert len(rows) == 3 * len(styles)
    sample = PairedDataset(manifest, "train", 24)[0]
    assert sample["valid_mask"].sum() == 16 * 18


def test_storage_guard_enforces_both_limits(tmp_path, monkeypatch):
    import shutil

    from photograder.acquisition import StorageGuard

    guard = StorageGuard(tmp_path, min_free=100, max_staging=1000)
    (tmp_path / "existing").write_bytes(b"a" * 100)
    usage_type = type(shutil.disk_usage(tmp_path))
    monkeypatch.setattr(shutil, "disk_usage", lambda root: usage_type(1000, 850, 150))
    guard.check(50)
    with pytest.raises(OSError, match="less than"):
        guard.check(51)
    with pytest.raises(OSError, match="staging"):
        guard.check(901)


def test_native_icc_transform_keeps_16bit_source_information(tmp_path):
    from pathlib import Path

    from photograder.acquisition import _verify_native_icc_precision

    source_profile = Path("/System/Library/ColorSync/Profiles/ROMM RGB.icc")
    target_profile = Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc")
    if not source_profile.exists() or not target_profile.exists() or not Path("/usr/bin/sips").exists():
        pytest.skip("Native ColorSync is only available on macOS")
    result = _verify_native_icc_precision(source_profile.read_bytes(), target_profile, tmp_path)
    assert result["native_rows_with_variation"] > 0
    assert result["prequantized_rows_with_variation"] == 0


def test_reused_path_checks_every_expected_digest(tmp_path):
    row = example(tmp_path)
    repeated = copy.deepcopy(row)
    repeated.update(style_id=1, style_name="Other style")
    repeated["sha256"]["input"] = "0" * 64
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([row, repeated], manifest)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_manifest(manifest)


def test_cross_split_alias_cannot_hide_behind_claimed_hash(tmp_path):
    row = example(tmp_path)
    alias = copy.deepcopy(row)
    (tmp_path / "alias_directory").mkdir()
    alias.update(
        scene_id="fixture:other",
        split="test",
        input="alias_directory/../" + row["input"],
        target="other_target.png",
    )
    Image.new("RGB", (40, 20), (13, 127, 254)).save(tmp_path / alias["target"])
    alias["sha256"] = {"input": "0" * 64, "target": sha256_file(tmp_path / alias["target"])}
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([row, alias], manifest)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_manifest(manifest)
    alias["sha256"]["input"] = row["sha256"]["input"]
    with pytest.raises(ValueError, match="different splits"):
        write_manifest([row, alias], tmp_path / "truthful.jsonl")
