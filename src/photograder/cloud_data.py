"""Prepare public FiveK pairs for a fresh scene-disjoint cloud training run.

No networking or cloud execution occurs here. The image cache is exactly the
existing PairedDataset 256px representation, encoded losslessly as uint8 NPZ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .acquisition import GIB, StorageGuard
from .data import PairedDataset, read_manifest, sha256_file, write_manifest

CACHE_VERSION = "paired-uint8-npz-v1"
SEED = 20260918


def split_scenes(scene_ids, seed=SEED, inner_val_fraction=0.0):
    """Round70% to nearest scene, then optionally hold validation inside train."""
    if not 0 <= inner_val_fraction < 1:
        raise ValueError("inner_val_fraction must be in [0,1)")
    scenes = sorted(set(scene_ids))
    if len(scenes) < 2:
        raise ValueError("At least two unique scenes are required")

    def rank(scene):
        return hashlib.sha256(f"cloud-full7030-v1:{seed}:{scene}".encode()).hexdigest()

    ordered = sorted(scenes, key=rank)
    count = max(1, min(len(scenes) - 1, int(math.floor(0.7 * len(scenes) + 0.5))))
    outer = {scene: "train" if index < count else "test" for index, scene in enumerate(ordered)}
    inner = dict(outer)
    validation_count = int(math.floor(count * inner_val_fraction + 0.5))
    if inner_val_fraction and not 0 < validation_count < count:
        raise ValueError("Inner validation fraction leaves an empty train or validation partition")

    def val_rank(scene):
        return hashlib.sha256(f"cloud-inner-val-v1:{seed}:{scene}".encode()).hexdigest()

    for scene in sorted(ordered[:count], key=val_rank)[:validation_count]:
        inner[scene] = "val"
    return outer, inner


def _oriented_size(path):
    with Image.open(path) as image:
        size = image.size
        return size[::-1] if image.getexif().get(274, 1) in (5, 6, 7, 8) else size


def _inspect_source_file(path):
    return sha256_file(path), _oriented_size(path)


def merge_public_pairs(
    manifests, output_directory, seed=SEED, inner_val_fraction=0.0, verification_workers=4
):
    """Verify public files with bounded I/O concurrency, then repartition.

    verification_workers controls filesystem resolution, hashing and headers;
    it does not change the separate image decoding/encoding worker limit.
    """
    if not isinstance(verification_workers, int) or not 1 <= verification_workers <= 64:
        raise ValueError("verification_workers must be an integer from 1 to 64")
    output_directory = Path(output_directory).resolve()
    collected, provenance, pending, source_paths = {}, [], [], {}
    for supplied_manifest in manifests:
        supplied_manifest = Path(supplied_manifest).resolve()
        provenance.append({"path": str(supplied_manifest), "sha256": sha256_file(supplied_manifest)})
        for original in read_manifest(supplied_manifest):
            if original["source_id"] != "MIT-Adobe-FiveK":
                raise ValueError("Only the explicitly acquired public MIT-Adobe-FiveK pairs are permitted")
            if (
                original["style_id"] not in (0, 1, 2)
                or original["style_name"]
                != {0: "FiveK Expert C", 1: "FiveK Expert A", 2: "FiveK Expert B"}[original["style_id"]]
            ):
                raise ValueError("Unexpected public FiveK style registry")
            row = dict(original)
            paths = {}
            for role in ("input", "target", "valid_mask"):
                if role not in row:
                    continue
                # Keep resolution in the worker pool: resolve() itself performs
                # expensive metadata reads on a cold remote filesystem.
                path = supplied_manifest.parent / row[role]
                paths[role] = path
                source_paths.setdefault(path, None)
            pending.append((row, paths))
    started = time.perf_counter()

    def progress(stage, completed, total):
        if completed == 0 or completed % 500 == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "source_verification": stage,
                        "completed_files": completed,
                        "total_files": total,
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                    }
                ),
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=verification_workers) as pool:
        progress("resolve", 0, len(source_paths))
        resolved = {}
        for count, (path, actual) in enumerate(
            zip(source_paths, pool.map(Path.resolve, source_paths), strict=False), 1
        ):
            resolved[path] = actual
            progress("resolve", count, len(source_paths))
        unique_paths = list(dict.fromkeys(resolved.values()))
        progress("checksum_and_header", 0, len(unique_paths))
        inspected = {}
        for count, (path, facts) in enumerate(
            zip(unique_paths, pool.map(_inspect_source_file, unique_paths), strict=False), 1
        ):
            inspected[path] = facts
            progress("checksum_and_header", count, len(unique_paths))
    # Assemble in original manifest order, independently of worker completion.
    for row, paths in pending:
        paths = {role: resolved[path] for role, path in paths.items()}
        for role, path in paths.items():
            # Shared paths reuse actual facts, never another row's validation.
            if inspected[path][0] != row["sha256"].get(role):
                raise ValueError(f"Actual source checksum mismatch: {path}")
            row[role] = os.path.relpath(path, output_directory)
        size = inspected[paths["input"]][1]
        if size != inspected[paths["target"]][1] or (
            "valid_mask" in paths and size != inspected[paths["valid_mask"]][1]
        ):
            raise ValueError(f"Source geometry mismatch: {row['scene_id']}")
        key = (row["scene_id"], row["style_id"])
        if key in collected:
            previous = collected[key]
            if any(
                row.get(field) != previous.get(field)
                for field in ("source_id", "sha256", "render_version", "split", "source_target_sha256")
            ):
                raise ValueError(f"Conflicting duplicate scene/style: {key}")
            continue
        collected[key] = row
    outer, partitions = split_scenes([key[0] for key in collected], seed, inner_val_fraction)
    rows = []
    for row in collected.values():
        row = dict(row)
        row["original_split"] = row["split"]
        row["outer_split"] = outer[row["scene_id"]]
        row["split"] = partitions[row["scene_id"]]
        rows.append(row)
    rows.sort(
        key=lambda row: ({"train": 0, "val": 1, "test": 2}[row["split"]], row["scene_id"], row["style_id"])
    )
    return rows, outer, provenance


def _quantize_sample(sample):
    result = {}
    for name in ("image", "target"):
        value = sample[name].numpy()
        # NumPy keeps CPU preparation bounded to the explicit worker pool;
        # large Torch elementwise thread pools are unnecessary for byte encoding.
        encoded = np.rint(value * np.float32(255)).astype(np.uint8)
        if not np.array_equal(encoded.astype(np.float32) / np.float32(255), value):
            raise ValueError("Cache uint8 round-trip would alter decoded training values")
        result[name] = encoded
    mask = sample["valid_mask"].numpy()
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError(
            "This public JPEG cache requires binary validity masks; fractional masks cannot be quantized"
        )
    result["valid_mask"] = mask.astype(np.uint8)
    for name in ("style_id", "original_size", "content_box"):
        result[name] = sample[name].numpy()
    result["scene_id"] = sample["scene_id"]
    return result


def prepare_cloud_data(
    manifests,
    output_directory,
    registry_path=None,
    seed=SEED,
    inner_val_fraction=0.0,
    workers=4,
    shard_size=32,
    max_cache_bytes=GIB,
    expected_scenes=None,
    expected_rows=None,
    verification_workers=4,
):
    started = time.perf_counter()
    if not 1 <= workers <= 4 or not 1 <= shard_size <= 64:
        raise ValueError("Use1..4 workers and1..64 examples per shard")
    if not isinstance(verification_workers, int) or not 1 <= verification_workers <= 64:
        raise ValueError("verification_workers must be an integer from 1 to 64")
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Cloud output must be a new empty directory; existing artifacts are immutable")
    if registry_path and Path(registry_path).exists():
        raise FileExistsError("Frozen cloud registry already exists; choose a new version")
    guard = StorageGuard(output, max_staging=min(20 * GIB, max_cache_bytes + 16 * 1024**2))
    guard.check()
    rows, outer, sources = merge_public_pairs(
        manifests, output, seed, inner_val_fraction, verification_workers=verification_workers
    )
    if expected_scenes is not None and len(outer) != expected_scenes:
        raise ValueError(f"Expected {expected_scenes} source scenes, found {len(outer)}")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} unique pairs, found {len(rows)}")
    manifest = output / "manifest.jsonl"
    write_manifest(rows, manifest)
    # Preserve exactly the existing data pipeline without modifying its source manifest.
    datasets = {
        split: PairedDataset(manifest, split, image_size=256)
        for split in ("train", "val", "test")
        if any(row["split"] == split for row in rows)
    }
    sequence = [(split, index) for split, dataset in datasets.items() for index in range(len(dataset))]

    def process(position):
        split, index = position
        return _quantize_sample(datasets[split][index])

    cache_bytes, shards, records = 0, [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(sequence), shard_size):
            positions = sequence[start : start + shard_size]
            samples = list(pool.map(process, positions))
            arrays = {
                name: np.stack([sample[name] for sample in samples])
                for name in ("image", "target", "valid_mask", "style_id", "original_size", "content_box")
            }
            arrays["scene_id"] = np.asarray([sample["scene_id"] for sample in samples], dtype=np.str_)
            uncompressed = sum(array.nbytes for array in arrays.values())
            guard.check(uncompressed + 65536)
            temporary = output / f"shard_{len(shards):05d}.partial"
            destination = temporary.with_suffix(".npz")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            size = temporary.stat().st_size
            if cache_bytes + size > max_cache_bytes:
                temporary.unlink()
                raise OSError("Compressed tensor cache would exceed configured1GiB budget")
            temporary.replace(destination)
            shard = {
                "path": destination.name,
                "sha256": sha256_file(destination),
                "bytes": size,
                "rows": len(samples),
            }
            shard_index = len(shards)
            shards.append(shard)
            records.extend(
                {"shard": shard_index, "offset": offset, "manifest_row": start + offset}
                for offset in range(len(samples))
            )
            cache_bytes += size
            if len(shards) % 10 == 0 or start + len(samples) == len(sequence):
                print(
                    json.dumps(
                        {
                            "prepared_rows": start + len(samples),
                            "total_rows": len(rows),
                            "cache_bytes": cache_bytes,
                        }
                    ),
                    flush=True,
                )
    index = {
        "cache_version": CACHE_VERSION,
        "image_size": 256,
        "manifest": manifest.name,
        "manifest_sha256": sha256_file(manifest),
        "requires_training_from_scratch": True,
        "seed": seed,
        "outer_train_fraction": 0.7,
        "rounding": "nearest whole scene, halves up",
        "outer_scene_counts": {
            split: sum(value == split for value in outer.values()) for split in ("train", "test")
        },
        "inner_val_fraction": inner_val_fraction,
        "scene_counts": {
            split: len({row["scene_id"] for row in rows if row["split"] == split})
            for split in ("train", "val", "test")
        },
        "row_counts": {
            split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")
        },
        "styles": {str(row["style_id"]): row["style_name"] for row in rows},
        "source_manifests": sources,
        "shards": shards,
        "records": records,
        "cache_bytes": cache_bytes,
        "preparation_seconds": time.perf_counter() - started,
        "representation": "ExistingPairedDataset EXIF/ICC->8bitRGB->LANCZOS aspect-preserving256canvas/edgepadding. uint8 divided by255 exactly recovers originalfloat32 values; binaryvalidity preserved. No new resampling or quantization.",
        "evaluation": "Fresh70/30 scene split invalidates old checkpoints; test must not select checkpoints. No inner validation unless explicitly configured.",
    }
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    if registry_path:
        registry = {
            "format_version": 1,
            "name": "cloud_full_7030",
            "seed": seed,
            "requires_training_from_scratch": True,
            "original_manifest_sha256": index["manifest_sha256"],
            "source_manifests": sources,
            "styles": index["styles"],
            "outer_splits": {
                split: sorted(scene for scene, part in outer.items() if part == split)
                for split in ("train", "test")
            },
            "splits": {
                split: list(dict.fromkeys(row["scene_id"] for row in rows if row["split"] == split))
                for split in ("train", "val", "test")
            },
            "inner_val_fraction": inner_val_fraction,
            "expected_rows": [
                {
                    key: row[key]
                    for key in (
                        "source_id",
                        "scene_id",
                        "style_id",
                        "split",
                        "outer_split",
                        "original_split",
                        "render_version",
                        "sha256",
                        "source_target_sha256",
                    )
                    if key in row
                }
                for row in rows
            ],
        }
        registry_path = Path(registry_path)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with registry_path.open("x") as stream:
            json.dump(registry, stream, indent=2)
            stream.write("\n")
    return index


class CachedPairedDataset(Dataset):
    """Lossless cloud cache; source JPEGs are not needed at load time.

    preload_to('cuda') keeps only compact uint8 arrays on GPU. iter_batches then
    selects indices and converts only each minibatch to float32. No NPZ decode
    happens during subsequent epochs. Without preload, one shard is held per
    dataset instance, providing a bounded-memory ordinary Dataset fallback.
    """

    def __init__(self, index_path, split, limit=None):
        self.index_path = Path(index_path).resolve()
        self.index = json.loads(self.index_path.read_text())
        if self.index.get("cache_version") != CACHE_VERSION:
            raise ValueError("Unsupported tensor cache version")
        self.image_size = int(self.index["image_size"])
        self.manifest_path = self.index_path.parent / self.index["manifest"]
        self.manifest_sha256 = sha256_file(self.manifest_path)
        if self.manifest_sha256 != self.index["manifest_sha256"]:
            raise ValueError("Cached manifest checksum mismatch")
        all_rows = read_manifest(self.manifest_path)
        records = self.index["records"]
        if len(records) != len(all_rows) or [record["manifest_row"] for record in records] != list(
            range(len(all_rows))
        ):
            raise ValueError("Cache index must map every manifest row exactly once in order")
        if split not in {"train", "val", "test"} or (limit is not None and limit < 1):
            raise ValueError("Invalid split or limit")
        chosen = [i for i, row in enumerate(all_rows) if row["split"] == split]
        if limit is not None:
            chosen = chosen[:limit]
        self.rows = [all_rows[i] for i in chosen]
        self.records = [records[i] for i in chosen]
        if not self.rows:
            raise ValueError(f"No cached {split} rows")
        self._loaded_shard = None
        self._arrays = None
        self._preloaded = None
        self._preloaded_device = None
        self._verified_shards = set()

        def verify_shard(number):
            metadata = self.index["shards"][number]
            if sha256_file(self.index_path.parent / metadata["path"]) != metadata["sha256"]:
                raise ValueError("Tensor shard checksum mismatch")
            return number

        # Hash streams are small; decoding still loads only one shard at a time.
        # Complete every selected-shard check before exposing a usable dataset.
        with ThreadPoolExecutor(max_workers=8) as pool:
            self._verified_shards.update(
                pool.map(verify_shard, sorted({record["shard"] for record in self.records}))
            )

    def __len__(self):
        return len(self.rows)

    def _load_shard(self, number):
        if self._loaded_shard != number:
            metadata = self.index["shards"][number]
            with np.load(self.index_path.parent / metadata["path"], allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            count, size = metadata["rows"], self.index["image_size"]
            for key, channels in (("image", 3), ("target", 3), ("valid_mask", 1)):
                if arrays[key].dtype != np.uint8 or arrays[key].shape != (count, channels, size, size):
                    raise ValueError("Invalid cached tensor shape or dtype")
            if not np.isin(arrays["valid_mask"], (0, 1)).all():
                raise ValueError("Invalid cached binary validity mask")
            self._arrays, self._loaded_shard = arrays, number
        return self._arrays

    def __getitem__(self, index):
        record = self.records[index]
        arrays = self._load_shard(record["shard"])
        offset = record["offset"]
        row = self.rows[index]
        if (
            str(arrays["scene_id"][offset]) != row["scene_id"]
            or int(arrays["style_id"][offset]) != row["style_id"]
        ):
            raise ValueError("Cached scene/style does not match manifest")
        return {
            "image": torch.from_numpy(arrays["image"][offset]).float().div(255),
            "target": torch.from_numpy(arrays["target"][offset]).float().div(255),
            "valid_mask": torch.from_numpy(arrays["valid_mask"][offset]).float(),
            "style_id": torch.tensor(int(arrays["style_id"][offset]), dtype=torch.long),
            "scene_id": row["scene_id"],
            "original_size": torch.from_numpy(arrays["original_size"][offset].copy()),
            "content_box": torch.from_numpy(arrays["content_box"][offset].copy()),
        }

    def preload_to(self, device):
        device = torch.device(device)
        if self._preloaded_device == device:
            return self
        size = self.index["image_size"]
        specs = {
            "image": ((3, size, size), torch.uint8),
            "target": ((3, size, size), torch.uint8),
            "valid_mask": ((1, size, size), torch.uint8),
            "style_id": ((), torch.long),
            "original_size": ((2,), torch.long),
            "content_box": ((4,), torch.long),
        }
        buffers = {
            key: torch.empty((len(self), *shape), dtype=dtype, device=device)
            for key, (shape, dtype) in specs.items()
        }
        groups = {}
        for i, record in enumerate(self.records):
            groups.setdefault(record["shard"], []).append((i, record["offset"]))
        for shard, positions in groups.items():
            arrays = self._load_shard(shard)
            destinations, offsets = zip(*positions, strict=False)
            destination_index = torch.tensor(destinations, dtype=torch.long, device=device)
            for i, offset in positions:
                if (
                    str(arrays["scene_id"][offset]) != self.rows[i]["scene_id"]
                    or int(arrays["style_id"][offset]) != self.rows[i]["style_id"]
                ):
                    raise ValueError("Cached scene/style does not match manifest")
            for key in specs:
                source = torch.from_numpy(arrays[key][list(offsets)].copy()).to(
                    device=device, dtype=specs[key][1]
                )
                buffers[key].index_copy_(0, destination_index, source)
        self._preloaded, self._preloaded_device = buffers, device
        self._loaded_shard, self._arrays = None, None
        return self

    def iter_batches(self, batch_size, shuffle, seed, device):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(self), generator=generator).tolist() if shuffle else list(range(len(self)))
        device = torch.device(device)
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            if self._preloaded is not None:
                indices = torch.tensor(positions, dtype=torch.long, device=self._preloaded_device)
                batch = {
                    key: tensor.index_select(0, indices).to(device) for key, tensor in self._preloaded.items()
                }
                for key in ("image", "target"):
                    batch[key] = batch[key].float().div(255)
                batch["valid_mask"] = batch["valid_mask"].float()
                batch["scene_id"] = [self.rows[i]["scene_id"] for i in positions]
            else:
                samples = [self[i] for i in positions]
                batch = {
                    key: torch.stack([sample[key] for sample in samples]).to(device)
                    for key in ("image", "target", "valid_mask", "style_id", "original_size", "content_box")
                }
                batch["scene_id"] = [sample["scene_id"] for sample in samples]
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fivek", default="data/fivek/manifest.jsonl")
    parser.add_argument("--multistyle", default="data/fivek_multistyle/manifest.jsonl")
    parser.add_argument("--output", default="data/cloud_full_7030")
    parser.add_argument("--registry", default="configs/splits/cloud_full_7030.json")
    parser.add_argument("--inner-val-fraction", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verification-workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = prepare_cloud_data(
        [args.fivek, args.multistyle],
        args.output,
        args.registry,
        inner_val_fraction=args.inner_val_fraction,
        workers=args.workers,
        verification_workers=args.verification_workers,
        expected_scenes=2748,
        expected_rows=2796,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "outer_scene_counts",
                    "scene_counts",
                    "row_counts",
                    "cache_bytes",
                    "preparation_seconds",
                    "manifest_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
