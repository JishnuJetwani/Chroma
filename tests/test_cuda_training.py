"""CPU-verifiable mechanics for the opt-in CUDA training path; no GPU required."""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import default_collate

from photograder import training
from photograder.data import RENDER_VERSION, PairedDataset, sha256_file, write_manifest
from photograder.models import ModelConfig, build_model
from photograder.runtime import load_checkpoint


@pytest.fixture
def train_only_manifest(tmp_path):
    rng = np.random.default_rng(62)
    rows = []
    for index in range(4):
        pixels = rng.integers(30, 220, (20, 24, 3), dtype=np.uint8)
        source, target = tmp_path / f"{index}-in.png", tmp_path / f"{index}-out.png"
        Image.fromarray(pixels).save(source)
        Image.fromarray((pixels * 0.72).astype(np.uint8)).save(target)
        rows.append(
            {
                "source_id": str(index),
                "scene_id": str(index),
                "input": source.name,
                "target": target.name,
                "style_id": 0,
                "style_name": "Expert C",
                "split": "train",
                "render_version": RENDER_VERSION,
                "sha256": {"input": sha256_file(source), "target": sha256_file(target)},
            }
        )
    path = tmp_path / "train-only.jsonl"
    write_manifest(rows, path)
    return path


def config(manifest, output, **options):
    return {
        "model": {"architecture": "fixed", "lut_size": 5},
        "data": {"manifest": str(manifest), "image_size": 32},
        "training": {
            "device": "cpu",
            "validation": False,
            "epochs": 2,
            "batch_size": 2,
            "accumulation_steps": 1,
            "learning_rate": 0.01,
            "cpu_threads": 2,
            "log_every": 0,
            **options,
        },
        "output_dir": str(output),
    }


def same_weights(left, right):
    a, _ = load_checkpoint(left)
    b, _ = load_checkpoint(right)
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)


def test_fixed_schedule_needs_no_validation_and_selects_final(train_only_manifest, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed schedule must never select using validation/test data")

    monkeypatch.setattr(training, "validation_loss", forbidden)
    callbacks = []
    output = tmp_path / "fixed"
    result = training.train(
        config(train_only_manifest, output),
        checkpoint_callback=lambda path, row: callbacks.append((path, row)),
    )
    assert result["status"] == "completed" and result["selected_by"] == "fixed_schedule_final"
    assert result["validation_samples"] == 0 and result["best_val_loss"] is None
    assert result["selected_step"] == 4 and result["final_selected"]
    assert len(callbacks) == 2 and callbacks[-1][1]["final_selected"]
    assert all(path == output for path, _ in callbacks)
    same_weights(output / "best.pt", output / "last.pt")
    same_weights(output / "final.pt", output / "last.pt")
    assert all(row["validation_l1"] is None for row in json.loads((output / "history.json").read_text()))
    with pytest.raises(ValueError, match="val"):
        training.train(config(train_only_manifest, tmp_path / "default-val", validation=True))


def test_time_stop_resumes_partial_epoch_at_optimizer_boundary(train_only_manifest, tmp_path, monkeypatch):
    full = tmp_path / "full"
    options = dict(epochs=1, batch_size=1, accumulation_steps=2)
    training.train(config(train_only_manifest, full, **options))
    clock = [0.0]
    original = training._loss

    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] += 1
        return result

    monkeypatch.setattr(training, "_loss", measured)
    monkeypatch.setattr(training.time, "perf_counter", lambda: clock[0])
    output = tmp_path / "interrupted"
    stopped = training.train(config(train_only_manifest, output, max_seconds=1.5, **options))
    assert stopped["status"] == "budget_stopped" and stopped["total_optimizer_steps"] == 1
    assert stopped["resume_cursor"] == {"epoch_index": 0, "next_batch_index": 2}
    assert stopped["best_checkpoint"] is None and stopped["final_checkpoint"] is None
    assert not (output / "final.pt").exists() and not (output / "best.pt").exists()
    resumed = training.train(config(train_only_manifest, output, **options), resume=output / "last.pt")
    assert resumed["status"] == "completed" and resumed["total_optimizer_steps"] == 2
    assert resumed["samples_processed_this_run"] == 2
    same_weights(full / "final.pt", output / "final.pt")


def test_logging_does_not_change_parameters(train_only_manifest, tmp_path, monkeypatch):
    calls = []
    original = training._log_components

    def capture(loss, components):
        assert all(
            isinstance(value, torch.Tensor) and not value.requires_grad for value in components.values()
        )
        calls.append(True)
        return original(loss, components)

    monkeypatch.setattr(training, "_log_components", capture)
    quiet, logged = tmp_path / "quiet", tmp_path / "logged"
    training.train(config(train_only_manifest, quiet))
    assert calls == []
    training.train(config(train_only_manifest, logged, log_every=1))
    assert len(calls) == 4
    same_weights(quiet / "final.pt", logged / "final.pt")


class TinyCache:
    """Implements the dataset cache contract without allocating a GPU."""

    def __init__(self, manifest):
        self.dataset = PairedDataset(manifest, "train", image_size=32)
        self.rows = self.dataset.rows
        self.manifest_sha256 = sha256_file(manifest)
        self.calls = []

    def __len__(self):
        return len(self.rows)

    def iter_batches(self, batch_size, shuffle, seed, device):
        self.calls.append((batch_size, shuffle, seed, str(device)))
        order = (
            torch.randperm(len(self), generator=torch.Generator().manual_seed(seed)).tolist()
            if shuffle
            else range(len(self))
        )
        for index in range(0, len(self), batch_size):
            yield default_collate([self.dataset[item] for item in order[index : index + batch_size]])


def test_cache_override_uses_direct_batches_and_preserves_loss_gradients(
    train_only_manifest, tmp_path, monkeypatch
):
    cache = TinyCache(train_only_manifest)
    direct = default_collate([cache.dataset[i] for i in range(2)])
    cached = next(cache.iter_batches(2, False, 42, torch.device("cpu")))
    torch.manual_seed(7)
    first = build_model(ModelConfig(architecture="fixed", lut_size=5))
    second = build_model(first.config)
    second.load_state_dict(first.state_dict())
    settings = {**training.DEFAULTS, "stage": "global"}
    loss_a, _ = training._loss(first, direct, settings, training.LOSS_DEFAULTS, 0)
    loss_b, terms = training._loss(second, cached, settings, training.LOSS_DEFAULTS, 0)
    assert all(isinstance(v, torch.Tensor) and not v.requires_grad for v in terms.values())
    loss_a.backward()
    loss_b.backward()
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    for a, b in zip(first.parameters(), second.parameters(), strict=False):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("iter_batches cache must bypass DataLoader")

    monkeypatch.setattr(training, "DataLoader", forbidden_loader)
    result = training.train(config(train_only_manifest, tmp_path / "cache"), train_dataset=cache)
    assert result["device_batch_cache"] and result["total_optimizer_steps"] == 4
    assert cache.calls[-2:] == [(2, True, 42, "cpu"), (2, True, 43, "cpu")]


def test_cuda_finite_check_uses_async_assert_without_scalar_read(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "_assert_async", lambda tensor, message: calls.append((tensor, message)))
    training._check_finite(torch.tensor(1.0), torch.device("cuda"), "finite-loss")
    assert len(calls) == 1 and calls[0][1] == "finite-loss"
    with pytest.raises(RuntimeError, match="bad-loss"):
        training._check_finite(torch.tensor(float("nan")), torch.device("cpu"), "bad-loss")


def test_cache_override_cannot_silently_change_rows(train_only_manifest, tmp_path):
    cache = TinyCache(train_only_manifest)
    cache.rows = list(reversed(cache.rows))
    with pytest.raises(ValueError, match="override rows differ"):
        training.train(config(train_only_manifest, tmp_path / "wrong-cache"), train_dataset=cache)


def test_benchmark_records_actual_accumulation_and_cached_throughput(train_only_manifest, tmp_path):
    cache = TinyCache(train_only_manifest)
    result = training.benchmark(
        config(train_only_manifest, tmp_path / "unused", batch_size=1, accumulation_steps=2),
        steps=3,
        output_dir=tmp_path / "benchmark",
        train_dataset=cache,
    )
    assert result["benchmark_version"] == "configured-microbatch-v2"
    assert result["step_unit"] == "microbatch"
    assert result["effective_batch_size"] == 2
    assert result["measured_samples"] == 3 and result["measured_optimizer_updates"] == 2
    assert result["samples_per_second"] > 0
    assert cache.calls and all(not row[1] for row in cache.calls)


def test_declared_update_budget_selects_final_even_mid_epoch(train_only_manifest, tmp_path):
    output = tmp_path / "update-budget"
    result = training.train(
        config(train_only_manifest, output, epochs=3, batch_size=1, accumulation_steps=2, max_steps=1)
    )
    assert result["status"] == "completed" and result["stopped_by_step_budget"]
    assert result["final_selected"] and result["total_optimizer_steps"] == 1
    assert result["resume_cursor"] == {"epoch_index": 0, "next_batch_index": 2}
    same_weights(output / "final.pt", output / "best.pt")


def test_real_uint8_cache_trains_without_original_images(train_only_manifest, tmp_path, monkeypatch):
    from photograder import cloud_data
    from photograder.acquisition import StorageGuard
    from photograder.data import read_manifest

    # Tiny synthetic fixtures do not consume the production disk reserve.
    monkeypatch.setattr(
        cloud_data, "StorageGuard", lambda root, **kwargs: StorageGuard(root, min_free=0, **kwargs)
    )
    rows = read_manifest(train_only_manifest)
    for row in rows:
        row.update(source_id="MIT-Adobe-FiveK", style_name="FiveK Expert C")
    write_manifest(rows, train_only_manifest)
    folder = tmp_path / "prepared"
    cloud_data.prepare_cloud_data([train_only_manifest], folder, workers=1, shard_size=2)
    dataset = cloud_data.CachedPairedDataset(folder / "index.json", "train").preload_to("cpu")
    # Prove train's cache path requires only packaged metadata and NPZ shards.
    for row in rows:
        for name in ("input", "target"):
            (train_only_manifest.parent / row[name]).unlink()
    cfg = config(folder / "manifest.jsonl", tmp_path / "cached-training", epochs=1)
    cfg["data"]["image_size"] = 256
    result = training.train(cfg, train_dataset=dataset)
    assert result["training_samples"] == len(dataset) == 3
    assert result["final_selected"] and result["validation_samples"] == 0
