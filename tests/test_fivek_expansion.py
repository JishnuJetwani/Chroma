import io
import json
import zipfile

import pytest
import torch
from PIL import Image
from test_cloud_data import make_sources

from photograder import cloud_data
from photograder.acquisition import StorageGuard, _member_bytes
from photograder.data import read_manifest, sha256_file, write_manifest
from photograder.fivek_expansion import _MemoryRange, combine_caches, grouped_ranges


def test_grouped_zip_ranges_preserve_exact_payloads():
    stream = io.BytesIO()
    values = {f"photos/{i}.jpg": bytes(range(256)) * (i + 1) for i in range(8)}
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in values.items():
            archive.writestr(name, value)
    data = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        groups = grouped_ranges(archive.infolist(), len(data))
        for start, end, members in groups:
            assert 0 <= start < end <= len(data)
            assert end - start <= 16 * 1024**2
            reader = _MemoryRange(start, data[start:end])
            for member in members:
                assert _member_bytes(reader, member) == values[member.filename]
    with pytest.raises(ValueError, match="window"):
        _MemoryRange(10, b"abc").range(9, 1)


def test_combine_preserves_splits_pixels_and_detects_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cloud_data, "StorageGuard", lambda root, **kwargs: StorageGuard(root, min_free=0, **kwargs)
    )
    roots = [tmp_path / "old", tmp_path / "extra"]
    for root in roots:
        root.mkdir()
    old_manifests = make_sources(roots[0], count=10)
    extra_manifest = make_sources(roots[1], count=10)[0]
    extra_rows = read_manifest(extra_manifest)
    for row in extra_rows:
        row["scene_id"] = row["scene_id"].replace("fixture", "new")
        for role in ("input", "target"):
            path = extra_manifest.parent / row[role]
            with Image.open(path) as image:
                flipped = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            flipped.save(path)
            row["sha256"][role] = sha256_file(path)
    write_manifest(extra_rows, extra_manifest)
    old_cache, extra_cache, combined = [tmp_path / name for name in ("old_cache", "extra_cache", "combined")]
    cloud_data.prepare_cloud_data(old_manifests, old_cache)
    cloud_data.prepare_cloud_data([extra_manifest], extra_cache)
    snapshots = {path: path.read_bytes() for cache in (old_cache, extra_cache) for path in cache.iterdir()}
    result = combine_caches(old_cache, extra_cache, combined, expected_scenes=20)
    assert result["outer_scene_counts"] == {"train": 14, "test": 6}
    assert sum(result["row_counts"].values()) == 24
    assert all(path.read_bytes() == before for path, before in snapshots.items())
    for split in ("train", "test"):
        actual = cloud_data.CachedPairedDataset(combined / "index.json", split)
        expected = {}
        for cache in (old_cache, extra_cache):
            source = cloud_data.CachedPairedDataset(cache / "index.json", split)
            for index, row in enumerate(source.rows):
                expected[(row["scene_id"], row["style_id"])] = source[index]
        for index, row in enumerate(actual.rows):
            source = expected.pop((row["scene_id"], row["style_id"]))
            for key in ("image", "target", "valid_mask"):
                assert torch.equal(actual[index][key], source[key])
        assert not expected
    with pytest.raises(ValueError, match="repeats"):
        combine_caches(old_cache, old_cache, tmp_path / "duplicate", expected_scenes=20)
    with pytest.raises(FileExistsError):
        combine_caches(old_cache, extra_cache, combined, expected_scenes=20)
    registry = json.loads((combined / "split_registry.json").read_text())
    assert not set(registry["outer_splits"]["train"]) & set(registry["outer_splits"]["test"])
    assert len(registry["cohorts"]["original"]) == len(registry["cohorts"]["additional"]) == 10


def test_training_requires_independently_frozen_protocol(tmp_path):
    from photograder.cloud_runner import SEED, STAGES, validate_cache_protocol

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("frozen manifest\n")
    digest = sha256_file(manifest)
    protocol = {
        "manifest_sha256": digest,
        "seed": SEED,
        "schedule": dict(STAGES),
        "scene_counts": {"train": 3500, "test": 1500},
        "row_counts": {"train": 3534, "val": 0, "test": 1514},
    }
    index = {
        "manifest_sha256": digest,
        "seed": SEED,
        "requires_training_from_scratch": True,
        "outer_scene_counts": protocol["scene_counts"],
        "inner_val_fraction": 0,
        "row_counts": protocol["row_counts"],
    }
    assert validate_cache_protocol(tmp_path, index, protocol) == protocol
    with pytest.raises(ValueError, match="Manifest"):
        validate_cache_protocol(tmp_path, index)
    with pytest.raises(ValueError, match="scene split"):
        validate_cache_protocol(
            tmp_path, {**index, "outer_scene_counts": {"train": 3501, "test": 1499}}, protocol
        )
    with pytest.raises(ValueError, match="schedule"):
        validate_cache_protocol(tmp_path, index, {**protocol, "schedule": {"joint": 100}})
    manifest.write_text("changed manifest\n")
    with pytest.raises(ValueError, match="Manifest"):
        validate_cache_protocol(tmp_path, index, protocol)
