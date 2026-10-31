"""Frozen-scene, cache-integrity and local-fault training protocol checks."""

import hashlib
import json

import numpy as np
import pytest
import torch

from photograder.raw_acquisition import INNER_VAL_COUNT, PARTITION_VERSION, SELECTION_VERSION, _rank
from photograder.raw_data import RENDER_VERSION
from photograder.raw_io import RAW_PIPELINE_VERSION
from photograder.raw_training import (
    TRUSTED_REGISTRY,
    _snapshot_sources,
    _training_generators,
    load_cache,
    local_fault,
    masked_l1,
    prepare_packed_cache,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_manifest(cache, rows):
    (cache / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def fixture_cache(tmp_path):
    registry_bytes = TRUSTED_REGISTRY.read_bytes()
    registry = json.loads(registry_bytes)
    train = registry["outer_splits"]["train"]
    val = set(sorted(train, key=lambda sid: _rank(sid, PARTITION_VERSION))[:INNER_VAL_COUNT])
    pools = {"train": set(train) - val, "val": val}
    records = []
    for split, count in (("train", 16), ("val", 8)):
        for scene_id in sorted(pools[split], key=lambda sid: _rank(sid, SELECTION_VERSION))[:count]:
            records.append({"scene_id": scene_id, "split": split, "outer_split": "train", "style_id": 0})
    plan = {
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "partition_version": PARTITION_VERSION,
        "selection_version": SELECTION_VERSION,
        "outer_test_used": False,
        "selected_counts": {"train": 16, "val": 8},
        "records": records,
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    rows = []
    for i, record in enumerate(records):
        metadata = {key: record[key] for key in ("scene_id", "split", "style_id")}
        metadata.update(
            render_version=RENDER_VERSION,
            training_size=256,
            content_box=[0, 32, 256, 224],
            raw={"pipeline_version": RAW_PIPELINE_VERSION},
            target={"source_bits": 16},
            geometry={"crop_method": "synthetic test fixture"},
        )
        image = np.full((3, 256, 256), 0.2, dtype=np.float32)
        image[0] = -0.05
        image[1] = 1.5
        target = np.full_like(image, 0.45)
        mask = np.zeros((1, 256, 256), dtype=np.float32)
        mask[:, 32:224] = 1
        path = tmp_path / f"{i}.npz"
        np.savez_compressed(
            path, image=image, target=target, valid_mask=mask, metadata=np.array(json.dumps(metadata))
        )
        rows.append({**metadata, "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_manifest(tmp_path, rows)
    yield tmp_path, rows, plan
    # Packed test tensors are roughly44MiB each; retain only the tiny NPZ test
    # fixtures in pytest's temporary artifacts between runs.
    for name in ("packed.pt", "packed.pt.partial"):
        (tmp_path / name).unlink(missing_ok=True)


def rewrite_sample(cache, rows, mutate, *, update_hash=True):
    path = cache / rows[0]["path"]
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key].copy() for key in data.files}
    mutate(values)
    np.savez_compressed(path, **values)
    if update_hash:
        rows[0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_manifest(cache, rows)


def test_cache_loads_unclipped_float_inputs_from_separate_inner_splits(fixture_cache):
    cache, _, _ = fixture_cache
    data, manifest_hash = load_cache(cache, "cpu")
    assert len(data["train"]["rows"]) == 16 and len(data["val"]["rows"]) == 8
    assert data["train"]["image"].min() < 0 and data["train"]["image"].max() > 1
    assert data["train"]["image"].dtype == data["val"]["target"].dtype == torch.float32
    assert manifest_hash == hashlib.sha256((cache / "manifest.jsonl").read_bytes()).hexdigest()
    assert not {row["scene_id"] for row in data["train"]["rows"]} & {
        row["scene_id"] for row in data["val"]["rows"]
    }


def test_plan_cannot_relabel_an_outer_test_scene_as_training(fixture_cache):
    cache, rows, plan = fixture_cache
    outer_test = json.loads(TRUSTED_REGISTRY.read_text())["outer_splits"]["test"][0]
    plan["records"][0]["scene_id"] = outer_test
    rows[0]["scene_id"] = outer_test
    (cache / "plan.json").write_text(json.dumps(plan))
    write_manifest(cache, rows)
    with pytest.raises(ValueError, match="frozen scene partition"):
        load_cache(cache, "cpu")


def test_plan_cannot_move_inner_validation_scene_to_training(fixture_cache):
    cache, rows, plan = fixture_cache
    plan["records"][0]["split"], plan["records"][16]["split"] = "val", "train"
    rows[0]["split"], rows[16]["split"] = "val", "train"
    (cache / "plan.json").write_text(json.dumps(plan))
    write_manifest(cache, rows)
    with pytest.raises(ValueError, match="frozen scene partition"):
        load_cache(cache, "cpu")


def test_plan_requires_trusted_registry_hash(fixture_cache):
    cache, _, plan = fixture_cache
    plan["registry_sha256"] = "0" * 64
    (cache / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="trusted frozen registry"):
        load_cache(cache, "cpu")


def test_manifest_duplicate_scene_rejected(fixture_cache):
    cache, rows, _ = fixture_cache
    rows[1]["scene_id"] = rows[0]["scene_id"]
    write_manifest(cache, rows)
    with pytest.raises(ValueError, match="Duplicate scene"):
        load_cache(cache, "cpu")


def test_cache_byte_tamper_rejected(fixture_cache):
    cache, rows, _ = fixture_cache
    rewrite_sample(cache, rows, lambda values: values["image"].fill(0.8), update_hash=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_cache(cache, "cpu")


def test_embedded_metadata_tamper_rejected_even_with_new_hash(fixture_cache):
    cache, rows, _ = fixture_cache

    def mutate(values):
        metadata = json.loads(str(values["metadata"]))
        metadata["split"] = "val"
        values["metadata"] = np.array(json.dumps(metadata))

    rewrite_sample(cache, rows, mutate)
    with pytest.raises(ValueError, match="Embedded RAW cache metadata"):
        load_cache(cache, "cpu")


@pytest.mark.parametrize("key", ["image", "target", "valid_mask"])
def test_all_arrays_require_float32(fixture_cache, key):
    cache, rows, _ = fixture_cache
    rewrite_sample(cache, rows, lambda values: values.__setitem__(key, values[key].astype(np.float64)))
    with pytest.raises(ValueError, match="Invalid float RAW pair"):
        load_cache(cache, "cpu")


@pytest.mark.parametrize("key", ["image", "target", "valid_mask"])
def test_nonfinite_cache_values_rejected(fixture_cache, key):
    cache, rows, _ = fixture_cache
    rewrite_sample(cache, rows, lambda values: values[key].__setitem__((0, 0, 0), np.nan))
    with pytest.raises(ValueError, match="Invalid float RAW pair"):
        load_cache(cache, "cpu")


def test_padding_mask_must_match_declared_content(fixture_cache):
    cache, rows, _ = fixture_cache
    rewrite_sample(cache, rows, lambda values: values["valid_mask"].__setitem__((0, 0, 0), 1))
    with pytest.raises(ValueError, match="content rectangle"):
        load_cache(cache, "cpu")


def test_cache_paths_cannot_escape_through_symlink(fixture_cache, tmp_path):
    cache, rows, _ = fixture_cache
    outside = tmp_path.parent / (tmp_path.name + "-outside.npz")
    outside.write_bytes((cache / rows[0]["path"]).read_bytes())
    link = cache / "outside-link.npz"
    link.symlink_to(outside)
    rows[0]["path"] = link.name
    write_manifest(cache, rows)
    with pytest.raises(ValueError, match="local to the manifest"):
        load_cache(cache, "cpu")


@pytest.mark.parametrize("shape", [(4, 3, 32, 128), (4, 3, 128, 32), (4, 3, 64, 64)])
@pytest.mark.parametrize("family", ["ellipse", "stripe"])
def test_fault_preserves_exact_outside_pixels_and_padding_for_all_aspects(shape, family):
    image = torch.linspace(-0.5, 3, int(np.prod(shape))).reshape(shape)
    valid = torch.ones_like(image[:, :1])
    valid[..., :3, :] = 0
    valid[..., -3:, :] = 0
    result, support = local_fault(image, torch.Generator().manual_seed(32), family=family, valid_mask=valid)
    assert result.shape == image.shape and support.shape == valid.shape
    assert torch.isfinite(result).all() and torch.isfinite(support).all()
    assert support.min() == 0 and support.max() <= 1
    outside = (support == 0).expand_as(image)
    assert torch.equal(result[outside], image[outside])
    assert torch.equal(support[valid == 0], torch.zeros_like(support[valid == 0]))
    assert (support > 0).any() and ((result != image) & ~outside).any()
    assert result.max() > 1 and result.min() < 0  # No display clipping was introduced.


def test_corruption_draws_do_not_change_matched_scene_sampling():
    clean_sampler, _ = _training_generators("cpu", 20260919)
    robust_sampler, robust_faults = _training_generators("cpu", 20260919)
    image = torch.ones(16, 3, 16, 24)
    for step in range(12):
        clean_indices = torch.randint(128, (16,), generator=clean_sampler)
        robust_indices = torch.randint(128, (16,), generator=robust_sampler)
        assert torch.equal(clean_indices, robust_indices)
        if step % 4 == 3:
            local_fault(image, robust_faults)


def test_masked_l1_excludes_padding_and_averages_channels():
    a, b = torch.zeros(1, 3, 4, 5), torch.ones(1, 3, 4, 5)
    mask = torch.zeros(1, 1, 4, 5)
    mask[:, :, 1:3, 1:4] = 1
    a[:, :, 0] = 999
    assert masked_l1(a, b, mask) == 1


def test_source_snapshot_preserves_exact_executing_code(tmp_path):
    hashes = _snapshot_sources(tmp_path)
    assert set(hashes) == {"raw_io.py", "raw_data.py", "raw_model.py", "raw_training.py"}
    for name, digest in hashes.items():
        assert hashlib.sha256((tmp_path / "source" / name).read_bytes()).hexdigest() == digest
    assert json.loads((tmp_path / "source/sha256.json").read_text()) == hashes


def test_cpu_pack_reads_each_npz_once_and_fast_load_avoids_npz(fixture_cache, monkeypatch):
    from pathlib import Path

    cache, rows, _ = fixture_cache
    original_read = Path.read_bytes
    reads = []

    def read(path):
        if path.suffix == ".npz":
            reads.append(path.name)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    receipt = prepare_packed_cache(cache)
    assert sorted(reads) == sorted(row["path"] for row in rows)
    assert receipt["counts"] == {"train": 16, "val": 8}
    assert receipt["packed_sha256"] == hashlib.sha256((cache / "packed.pt").read_bytes()).hexdigest()
    reads.clear()
    data, digest = load_cache(cache, "cpu")
    assert reads == []
    assert digest == receipt["manifest_sha256"]
    assert data["train"]["image"].min() < 0 and data["train"]["image"].max() > 1
    assert data["train"]["rows"] == rows[:16]


def test_packed_byte_tamper_is_rejected_without_npz_fallback(fixture_cache):
    cache, _, _ = fixture_cache
    prepare_packed_cache(cache)
    path = cache / "packed.pt"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        last = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([last[0] ^ 1]))
    with pytest.raises(ValueError, match="Packed RAW cache checksum"):
        load_cache(cache, "cpu")


def test_packed_receipt_is_invalidated_when_manifest_changes(fixture_cache):
    cache, _, _ = fixture_cache
    prepare_packed_cache(cache)
    with (cache / "manifest.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Packed RAW receipt differs"):
        load_cache(cache, "cpu")


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("shape", "tensor shape"),
        ("metadata", "rows differ"),
        ("dtype", "Invalid float RAW pair"),
        ("nonfinite", "Invalid float RAW pair"),
    ],
)
def test_packed_rehashed_tamper_still_rechecks_schema_and_values(fixture_cache, mutation, error):
    cache, _, _ = fixture_cache
    prepare_packed_cache(cache)
    path = cache / "packed.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    train = saved["splits"]["train"]
    if mutation == "shape":
        train["image"] = train["image"][:, :, :10].clone()
    elif mutation == "metadata":
        train["rows"][0]["split"] = "val"
    elif mutation == "dtype":
        train["target"] = train["target"].double()
    else:
        train["image"][0, 0, 0, 0] = float("nan")
    torch.save(saved, path)
    receipt_path = cache / "packed.source.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(
        packed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), packed_bytes=path.stat().st_size
    )
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=error):
        load_cache(cache, "cpu")


def test_pack_normalizes_only_display_target_roundoff(fixture_cache):
    cache, rows, _ = fixture_cache
    rewrite_sample(cache, rows, lambda values: values["target"].__setitem__((0, 0, 0), 1 + 5e-7))
    prepare_packed_cache(cache)
    data, _ = load_cache(cache, "cpu")
    assert data["train"]["target"][0, 0, 0, 0] == 1
    assert data["train"]["image"][0, 1, 0, 0] == 1.5
    assert data["train"]["image"][0, 0, 0, 0] < 0
