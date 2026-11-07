"""Extend the frozen FiveK experiment without moving any existing scene.

Only previously missing JPEG pairs are downloaded. Existing lossless tensor
shards are reused by reference, so the original experiment remains immutable.
"""

from __future__ import annotations

import binascii
import copy
import json
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .acquisition import (
    FIVEK_SIZE,
    FIVEK_URL,
    GIB,
    PREFIX,
    RemoteZipReader,
    StorageGuard,
    _member_bytes,
    _retain_fivek_terms,
)
from .cloud_data import CACHE_VERSION, SEED, CachedPairedDataset, prepare_cloud_data, split_scenes
from .data import read_manifest, sha256_file, write_manifest
from .runtime import atomic_json


def grouped_ranges(members, archive_size, max_range=16 * 1024**2, max_gap=256 * 1024):
    """Coalesce nearby ZIP entries while allowing maximal local-header fields."""
    groups = []
    for item in sorted(members, key=lambda entry: entry.header_offset):
        start = item.header_offset
        end = min(archive_size, start + 30 + 2 * 65535 + item.compress_size)
        if end - start > max_range:
            raise ValueError("ZIP member cannot fit in a bounded range")
        if groups and start - groups[-1][1] <= max_gap and end - groups[-1][0] <= max_range:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append(item)
        else:
            groups.append([start, end, [item]])
    return groups


class _MemoryRange:
    def __init__(self, start, data):
        self.start, self.data = start, data

    def range(self, start, size):
        offset = start - self.start
        if offset < 0 or offset + size > len(self.data):
            raise ValueError("ZIP local header exceeds its bounded download window")
        return self.data[offset : offset + size]


def acquire_missing(old_cache, output, *, workers=8, expected_total=5000, commit=lambda: None):
    """Fetch only missing public scene pairs, checking ZIP CRC and image hashes."""
    if not 1 <= workers <= 16:
        raise ValueError("Download workers must be 1..16")
    started = time.perf_counter()
    old_cache, output = Path(old_cache), Path(output)
    old_index = json.loads((old_cache / "index.json").read_text())
    old_manifest = old_cache / old_index["manifest"]
    if sha256_file(old_manifest) != old_index["manifest_sha256"]:
        raise ValueError("Original frozen manifest checksum mismatch")
    old_rows = read_manifest(old_manifest)
    old_ids = {row["scene_id"].removeprefix("fivek:") for row in old_rows}
    guard = StorageGuard(output)
    guard.check()
    reader = RemoteZipReader(FIVEK_URL, FIVEK_SIZE, byte_budget=16 * 1024**2)
    archive = zipfile.ZipFile(reader)
    entries = {}
    for role in ("input", "expertC"):
        prefix = f"{PREFIX}{role}/JPG/480p/"
        entries[role] = {
            Path(item.filename).stem: item
            for item in archive.infolist()
            if item.filename.startswith(prefix) and item.filename.endswith(".jpg")
        }
    ids = set(entries["input"])
    if ids != set(entries["expertC"]) or len(ids) != expected_total or not old_ids <= ids:
        raise ValueError("Archive does not contain the complete matching collection")
    missing = sorted(ids - old_ids)
    jobs, destinations = [], {}
    required_bytes = 0
    for photo_id in missing:
        for role in ("input", "expertC"):
            item = entries[role][photo_id]
            destination = output / role / f"{photo_id}.jpg"
            destinations[item.filename] = destination
            if destination.exists():
                payload = destination.read_bytes()
                if len(payload) != item.file_size or binascii.crc32(payload) & 0xFFFFFFFF != item.CRC:
                    raise ValueError(f"Previously downloaded photo failed CRC: {destination}")
            else:
                required_bytes += item.file_size
                jobs.append(item)
    guard.check(required_bytes)
    groups = grouped_ranges(jobs, FIVEK_SIZE)
    planned_bytes = sum(end - start for start, end, _ in groups)
    if planned_bytes > 3 * GIB:
        raise ValueError("Missing-pair request would exceed the 3 GiB network allowance")
    print(
        json.dumps(
            {
                "acquisition": "missing_pairs",
                "scenes": len(missing),
                "files_to_download": len(jobs),
                "range_requests": len(groups),
                "planned_network_bytes": planned_bytes,
            }
        ),
        flush=True,
    )
    local = threading.local()

    def fetch(group):
        if not hasattr(local, "reader"):
            local.reader = RemoteZipReader(FIVEK_URL, FIVEK_SIZE, byte_budget=3 * GIB)
        start, end, members = group
        payload = local.reader.range(start, end - start)
        if local.reader.last_modified != reader.last_modified:
            raise ValueError("Archive changed after inventory")
        memory = _MemoryRange(start, payload)
        for item in members:
            content = _member_bytes(memory, item)
            destination = destinations[item.filename]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".partial")
            temporary.write_bytes(content)
            temporary.replace(destination)
        return len(members), len(payload)

    completed, network_bytes = 0, reader.transferred
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, (count, transferred) in enumerate(pool.map(fetch, groups), 1):
            completed += count
            network_bytes += transferred
            if number % 10 == 0 or number == len(groups):
                print(
                    json.dumps(
                        {
                            "downloaded_files": completed,
                            "total_files": len(jobs),
                            "network_bytes": network_bytes,
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
            if number % 40 == 0:
                commit()
    _, assignments = split_scenes([f"fivek:{photo_id}" for photo_id in missing], SEED)
    rows = []
    for photo_id in missing:
        scene = f"fivek:{photo_id}"
        paths = {"input": f"input/{photo_id}.jpg", "target": f"expertC/{photo_id}.jpg"}
        rows.append(
            {
                "source_id": "MIT-Adobe-FiveK",
                "scene_id": scene,
                "style_id": 0,
                "style_name": "FiveK Expert C",
                "split": assignments[scene],
                "render_version": "HuiZeng-480p-srgb-jpeg-exif-v1",
                **paths,
                "sha256": {role: sha256_file(output / path) for role, path in paths.items()},
            }
        )
    write_manifest(rows, output / "manifest.jsonl")
    _retain_fivek_terms(output)
    report = {
        "dataset": "MIT-Adobe FiveK",
        "source": FIVEK_URL,
        "archive_bytes": FIVEK_SIZE,
        "archive_last_modified": reader.last_modified,
        "original_manifest_sha256": old_index["manifest_sha256"],
        "additional_scenes": len(rows),
        "archive_paired_scenes": len(ids),
        "manifest_sha256": sha256_file(output / "manifest.jsonl"),
        "network_bytes_this_attempt": network_bytes,
        "acquisition_seconds": time.perf_counter() - started,
        "integrity": "ZIP CRC32 and size for every file; SHA256 in manifest; geometry checked during caching",
    }
    atomic_json(output / "provenance.json", report)
    commit()
    return report


def combine_caches(old_cache, additional_cache, output, *, expected_scenes=5000):
    """Combine immutable caches, preserving assignments and tensor bytes."""
    old_cache, additional_cache, output = map(Path, (old_cache, additional_cache, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Expanded cache must be a new directory")
    collected, shards, sources, seen_scenes = [], [], [], set()
    base = None
    for source_number, cache in enumerate((old_cache, additional_cache)):
        index = json.loads((cache / "index.json").read_text())
        if (
            index["cache_version"] != CACHE_VERSION
            or index["image_size"] != 256
            or index["seed"] != SEED
            or index["inner_val_fraction"] != 0
        ):
            raise ValueError("Incompatible cache protocol")
        manifest = cache / index["manifest"]
        if sha256_file(manifest) != index["manifest_sha256"]:
            raise ValueError("Source cache manifest checksum mismatch")
        rows = read_manifest(manifest)
        scenes = {row["scene_id"] for row in rows}
        if seen_scenes & scenes:
            raise ValueError("Additional cache repeats existing scenes")
        seen_scenes.update(scenes)
        if len(rows) != len(index["records"]) or [r["manifest_row"] for r in index["records"]] != list(
            range(len(rows))
        ):
            raise ValueError("Source cache record mapping is invalid")
        for split in ("train", "test"):
            CachedPairedDataset(cache / "index.json", split)  # Verify every selected shard hash.
        shard_offset = len(shards)
        shards.extend(
            {**item, "path": os.path.relpath(cache / item["path"], output)} for item in index["shards"]
        )
        for row, record in zip(rows, index["records"], strict=True):
            row = copy.deepcopy(row)
            for role in ("input", "target", "valid_mask"):
                if role in row:
                    row[role] = os.path.relpath(cache / row[role], output)
            row["expansion_cohort"] = "original" if source_number == 0 else "additional"
            collected.append((row, {**record, "shard": record["shard"] + shard_offset}))
        sources.append(
            {
                "index": os.path.relpath(cache / "index.json", output),
                "index_sha256": sha256_file(cache / "index.json"),
                "manifest_sha256": index["manifest_sha256"],
            }
        )
        if base is None:
            base = copy.deepcopy(index)
    if len(seen_scenes) != expected_scenes:
        raise ValueError("Expanded cache scene count mismatch")
    collected.sort(
        key=lambda pair: (0 if pair[0]["split"] == "train" else 1, pair[0]["scene_id"], pair[0]["style_id"])
    )
    rows = [row for row, _ in collected]
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, output / "manifest.jsonl")
    # Includes identical-content leakage detection across both source caches.
    read_manifest(output / "manifest.jsonl")
    scene_counts = {
        split: len({row["scene_id"] for row in rows if row["split"] == split}) for split in ("train", "test")
    }
    if scene_counts["train"] != round(0.7 * expected_scenes):
        raise ValueError("Combined assignment is not the requested 70/30 partition")
    base.update(
        manifest="manifest.jsonl",
        manifest_sha256=sha256_file(output / "manifest.jsonl"),
        outer_scene_counts=scene_counts,
        scene_counts={**scene_counts, "val": 0},
        row_counts={split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")},
        shards=shards,
        records=[{**record, "manifest_row": i} for i, (_, record) in enumerate(collected)],
        cache_bytes=sum(item["bytes"] for item in shards),
        source_caches=sources,
        source_manifests=[{"path": item["index"], "sha256": item["index_sha256"]} for item in sources],
        preparation_seconds=None,
        split_policy="Preserve every existing assignment; SHA256-rank only additional scenes with the original seed",
        evaluation="Fixed-schedule fresh training; previous model evaluated on the identical expanded test set",
    )
    atomic_json(output / "index.json", base)
    registry = {
        "format_version": 1,
        "name": "fivek_5000_7030",
        "seed": SEED,
        "original_manifest_sha256": base["manifest_sha256"],
        "source_caches": sources,
        "outer_splits": {
            split: sorted({r["scene_id"] for r in rows if r["split"] == split}) for split in ("train", "test")
        },
        "cohorts": {
            cohort: sorted({r["scene_id"] for r in rows if r["expansion_cohort"] == cohort})
            for cohort in ("original", "additional")
        },
        "expected_rows": rows,
    }
    atomic_json(output / "split_registry.json", registry)
    return base


def prepare_expansion(data_root, *, commit=lambda: None):
    import torch

    torch.set_num_threads(1)
    root = Path(data_root)
    old = root / "cloud_full_7030_io32"
    additions = root / "fivek_5000_additional_pairs_v1"
    extra_cache = root / "fivek_5000_additional_cache_v1"
    output = root / "fivek_5000_7030_v1"
    if (output / "index.json").exists():
        raise FileExistsError("Expanded cache already exists; reuse its frozen protocol")
    acquisition = acquire_missing(old, additions, commit=commit)
    if not (extra_cache / "index.json").exists():
        prepare_cloud_data(
            [additions / "manifest.jsonl"],
            extra_cache,
            extra_cache / "split_registry.json",
            workers=4,
            expected_scenes=2252,
            expected_rows=2252,
            verification_workers=32,
        )
        commit()
    index = combine_caches(old, extra_cache, output)
    if index["outer_scene_counts"] != {"train": 3500, "test": 1500} or index["row_counts"] != {
        "train": 3534,
        "val": 0,
        "test": 1514,
    }:
        raise ValueError("Expanded run does not match the frozen scene/editor counts")
    protocol = {
        "manifest_sha256": index["manifest_sha256"],
        "scene_counts": index["outer_scene_counts"],
        "row_counts": index["row_counts"],
        "cache": output.name,
        "seed": SEED,
        "parent_manifest_sha256": index["source_caches"][0]["manifest_sha256"],
        "split_policy": index["split_policy"],
        "schedule": {"global": 10, "repair": 5, "joint": 10},
        "acquisition": acquisition,
    }
    atomic_json(output / "protocol.json", protocol)
    commit()
    print(json.dumps({"prepared_expansion": protocol}), flush=True)
    return protocol
