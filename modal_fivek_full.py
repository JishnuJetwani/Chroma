"""Acquire full FiveK on CPU, then train/evaluate a bounded H100 experiment."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("chroma-fivek-full")
data_volume = modal.Volume.from_name("selective-photograder-data-v1")
run_volume = modal.Volume.from_name("selective-photograder-runs-v1")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements-cloud.txt"), extra_options="--require-hashes")
    .env({"PYTHONPATH": "/root/project/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "src", "/root/project/src", ignore=["**/__pycache__/**"])
)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=2400,
    scaledown_window=2,
    max_containers=1,
    volumes={"/data": data_volume},
)
def prepare():
    from photograder.fivek_expansion import prepare_expansion

    return prepare_expansion("/data", commit=data_volume.commit)


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=16384,
    timeout=1500,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": data_volume.with_mount_options(read_only=True), "/runs": run_volume},
)
def train_full(run_id: str, protocol: dict, resume: bool = False):
    from photograder.cloud_runner import run

    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Run ID must be a single directory name")
    if protocol["scene_counts"] != {"train": 3500, "test": 1500}:
        raise ValueError("Expected complete FiveK 70/30 protocol")
    return run(
        Path("/data/fivek_5000_7030_v1"),
        Path("/runs") / run_id,
        budget_seconds=1200,
        commit=run_volume.commit,
        resume=resume,
        expected_protocol=protocol,
        baseline_checkpoint="/runs/full-7030-20260918/joint/model.pt",
        baseline_sha256="1e60aaab4c95a9c3755a6d6f314c4c531b355d984dc8edab5c8f776c5e8d657a",
    )


@app.local_entrypoint()
def main(train: bool = False, run_id: str = "fivek-5000-7030-20261107", resume: bool = False):
    import json

    destination = ROOT / "runs/fivek_full_20261107"
    destination.mkdir(parents=True, exist_ok=True)
    if train:
        protocol = json.loads((destination / "protocol.json").read_text())
        protocol["runtime"] = {"batch_size": 8, "encoder_autocast": "none"}
        (destination / "training_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        result = train_full.remote(run_id, protocol, resume)
        (destination / "remote_result.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = prepare.remote()
        (destination / "protocol.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
