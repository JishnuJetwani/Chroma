import copy
import json
import os
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageCms

from photograder import cloud_data
from photograder.acquisition import StorageGuard
from photograder.data import PairedDataset, read_manifest, sha256_file, write_manifest


def make_sources(tmp_path, count=10):
    root = tmp_path / "original"
    extra = tmp_path / "extra"
    root.mkdir()
    extra.mkdir()
    rows = []
    yy, xx = np.mgrid[:180, :300]
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    for number in range(count):
        image = np.stack(
            ((xx + number * 17) % 256, (yy + number * 23) % 256, (xx // 2 + yy + number * 31) % 256), axis=-1
        ).astype(np.uint8)
        paths = {"input": f"input_{number}.png", "target": f"c_{number}.png"}
        Image.fromarray(image).save(root / paths["input"], icc_profile=icc)
        Image.fromarray(np.roll(image, 1, axis=-1)).save(root / paths["target"], icc_profile=icc)
        row = {
            "source_id": "MIT-Adobe-FiveK",
            "scene_id": f"fivek:fixture{number:03d}",
            "style_id": 0,
            "style_name": "FiveK Expert C",
            "split": "train",
            "render_version": "fixture-srgb-v1",
            **paths,
        }
        row["sha256"] = {key: sha256_file(root / name) for key, name in paths.items()}
        rows.append(row)
    full = root / "manifest.jsonl"
    write_manifest(rows, full)
    extras = []
    for row in rows[:2]:
        duplicate = copy.deepcopy(row)
        for key in ("input", "target"):
            duplicate[key] = os.path.relpath(root / row[key], extra)
        extras.append(duplicate)
        for style in (1, 2):
            styled = copy.deepcopy(duplicate)
            target = extra / f"{row['scene_id'].replace(':', '_')}_{style}.png"
            with Image.open(root / row["target"]) as source:
                values = np.asarray(source).copy()
            values[..., 0] = (values[..., 0].astype(np.uint16) + style * 11) % 256
            Image.fromarray(values).save(target, icc_profile=icc)
            styled.update(
                style_id=style,
                style_name={1: "FiveK Expert A", 2: "FiveK Expert B"}[style],
                target=target.name,
                render_version="fixture-native-v1",
            )
            styled["sha256"]["target"] = sha256_file(target)
            extras.append(styled)
    additional = extra / "manifest.jsonl"
    write_manifest(extras, additional)
    return full, additional


@pytest.fixture
def fixture_storage(monkeypatch):
    # Synthetic unit fixtures must not depend on this host's current disk reserve.
    # Production still constructs the normal20GiB-reserve StorageGuard unchanged.
    monkeypatch.setattr(
        cloud_data, "StorageGuard", lambda root, **kwargs: StorageGuard(root, min_free=0, **kwargs)
    )


def test_scene_split_exact_counts_seed_and_optional_inner_val():
    ids = [f"fivek:a{number:04d}" for number in range(2748)]
    outer, split = cloud_data.split_scenes(ids)
    assert sum(value == "train" for value in outer.values()) == 1924
    assert sum(value == "test" for value in outer.values()) == 824
    assert split == outer
    assert cloud_data.split_scenes(reversed(ids))[0] == outer
    assert cloud_data.split_scenes(ids, seed=12)[0] != outer
    heldout, inner = cloud_data.split_scenes(ids, inner_val_fraction=0.1)
    assert heldout == outer
    assert sum(value == "val" for value in inner.values()) == 192
    assert all(outer[scene] == "train" for scene, part in inner.items() if part == "val")


def test_merge_dedup_and_all_styles_stay_with_scene(tmp_path):
    manifests = make_sources(tmp_path)
    originals = [path.read_bytes() for path in manifests]
    rows, outer, provenance = cloud_data.merge_public_pairs(manifests, tmp_path / "cloud")
    assert len(rows) == 14  # Ten C pairs plus four A/B pairs; two C duplicates removed.
    assert len(outer) == 10
    scene_parts = {}
    for row in rows:
        scene_parts.setdefault(row["scene_id"], set()).add(row["split"])
        assert row["original_split"] == "train"
    assert all(len(parts) == 1 for parts in scene_parts.values())
    assert all(path.read_bytes() == original for path, original in zip(manifests, originals, strict=False))
    assert len(provenance) == 2


def test_cache_matches_original_pipeline_and_gpu_batch_protocol(tmp_path, fixture_storage):
    manifests = make_sources(tmp_path, count=6)
    source_bytes = [path.read_bytes() for path in manifests]
    output = tmp_path / "cloud"
    registry = tmp_path / "split_registry.json"
    result = cloud_data.prepare_cloud_data(manifests, output, registry, workers=2, shard_size=3)
    assert result["row_counts"]["val"] == 0
    assert result["requires_training_from_scratch"]
    assert result["cache_bytes"] < 5 * 1024**2
    for split in ("train", "test"):
        expected = PairedDataset(output / "manifest.jsonl", split, image_size=256)
        actual = cloud_data.CachedPairedDataset(output / "index.json", split)
        assert actual.rows == expected.rows
        for index in range(len(actual)):
            for key in ("image", "target", "valid_mask", "style_id", "original_size", "content_box"):
                assert torch.equal(actual[index][key], expected[index][key]), key
        actual.preload_to("cpu")
        batches = list(actual.iter_batches(2, shuffle=False, seed=10, device="cpu"))
        assert torch.equal(
            torch.cat([batch["image"] for batch in batches]),
            torch.stack([expected[index]["image"] for index in range(len(expected))]),
        )
        first = [scene for batch in actual.iter_batches(2, True, 123, "cpu") for scene in batch["scene_id"]]
        second = [scene for batch in actual.iter_batches(2, True, 123, "cpu") for scene in batch["scene_id"]]
        assert first == second
    assert all(path.read_bytes() == original for path, original in zip(manifests, source_bytes, strict=False))
    with pytest.raises(FileExistsError):
        cloud_data.prepare_cloud_data(manifests, output)
    assert json.loads(registry.read_text())["original_manifest_sha256"] == sha256_file(
        output / "manifest.jsonl"
    )


def test_cache_rejects_source_and_cached_corruption(tmp_path, fixture_storage):
    manifests = make_sources(tmp_path, count=4)
    output = tmp_path / "cloud"
    result = cloud_data.prepare_cloud_data(manifests, output, workers=1, shard_size=3)
    shard = output / result["shards"][0]["path"]
    shard.write_bytes(b"corrupted archive")
    with pytest.raises(ValueError, match="shard checksum"):
        cloud_data.CachedPairedDataset(output / "index.json", "train")
    source_rows = read_manifest(manifests[0])
    source_rows[0]["source_id"] = "private-photos"
    write_manifest(source_rows, manifests[0])
    with pytest.raises(ValueError, match="public"):
        cloud_data.merge_public_pairs(manifests, tmp_path / "other")


def test_relocation_keeps_manifest_bytes_identical(tmp_path):
    import shutil

    first = tmp_path / "mac"
    second = tmp_path / "modal"
    first.mkdir()
    second.mkdir()
    sources = make_sources(first, count=4)
    for directory in ("original", "extra"):
        shutil.copytree(first / directory, second / directory)
    original, _, _ = cloud_data.merge_public_pairs(sources, first / "cloud")
    relocated, _, _ = cloud_data.merge_public_pairs(
        [second / "original" / "manifest.jsonl", second / "extra" / "manifest.jsonl"], second / "cloud"
    )
    assert original == relocated


def test_source_verification_pools_path_resolution_and_checks_unique_files(tmp_path, monkeypatch):
    manifests = make_sources(tmp_path, count=4)
    real_resolve = Path.resolve
    real_inspect = cloud_data._inspect_source_file
    inspected, workers = Counter(), set()
    lock = threading.Lock()

    def resolve(path, *args, **kwargs):
        if path.suffix == ".png":
            assert threading.current_thread() is not threading.main_thread()
        return real_resolve(path, *args, **kwargs)

    def inspect(path):
        with lock:
            inspected[path] += 1
            workers.add(threading.get_ident())
        return real_inspect(path)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(cloud_data, "_inspect_source_file", inspect)
    cloud_data.merge_public_pairs(manifests, tmp_path / "cloud")
    assert len(inspected) == 12  # Eight C files plus four extra targets.
    assert set(inspected.values()) == {1}  # Aliases and repeated style inputs share checks.
    assert 1 <= len(workers) <= 4
    assert threading.get_ident() not in workers

    extra_rows = read_manifest(manifests[1])
    extra_rows[1]["sha256"]["input"] = "0" * 64
    write_manifest(extra_rows, manifests[1])
    with pytest.raises(ValueError, match="source checksum mismatch"):
        cloud_data.merge_public_pairs(manifests, tmp_path / "rejected")


def test_verification_concurrency_is_separate_and_bounded(tmp_path, monkeypatch, fixture_storage):
    manifests = make_sources(tmp_path, count=2)
    real_executor = cloud_data.ThreadPoolExecutor
    worker_limits = []

    def executor(*args, **kwargs):
        worker_limits.append(kwargs["max_workers"])
        return real_executor(*args, **kwargs)

    monkeypatch.setattr(cloud_data, "ThreadPoolExecutor", executor)
    cloud_data.prepare_cloud_data(manifests, tmp_path / "cloud", verification_workers=32)
    assert worker_limits == [32, 4]  # Remote I/O concurrency never changes CPU decoding.
    cached = cloud_data.CachedPairedDataset(tmp_path / "cloud" / "index.json", "train")
    assert worker_limits == [32, 4, 8]
    assert cached._verified_shards == {record["shard"] for record in cached.records}
    assert cached._arrays is None  # Parallel verification does not decode NPZ tensors.
    for invalid in (0, 65, 2.5):
        with pytest.raises(ValueError, match="verification_workers"):
            cloud_data.merge_public_pairs(manifests, tmp_path / "bad", verification_workers=invalid)
        with pytest.raises(ValueError, match="verification_workers"):
            cloud_data.prepare_cloud_data(manifests, tmp_path / "bad", verification_workers=invalid)
    assert not (tmp_path / "bad").exists()
