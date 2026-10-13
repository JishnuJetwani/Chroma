"""Modal entrypoint: CPU image build, pre-uploaded cache, one bounded H100 run.

Usage: .venv/bin/modal run --detach modal_train.py --run-id <unique-name>
The cache must first be uploaded with scripts/upload_modal_data.py.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
DATA_VOLUME = "selective-photograder-data-v1"
RUN_VOLUME = "selective-photograder-runs-v1"
app = modal.App("selective-photograder-training")
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
run_volume = modal.Volume.from_name(RUN_VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements-cloud.txt"), extra_options="--require-hashes")
    .env({"PYTHONPATH": "/root/project/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "src", "/root/project/src", ignore=["**/__pycache__/**"])
)


@app.function(image=image, cpu=1, memory=2048, timeout=300, scaledown_window=2)
def preflight():
    import torch

    from photograder.models import ModelConfig, build_model

    model = build_model(ModelConfig(styles=("FiveK Expert C", "FiveK Expert A", "FiveK Expert B")))
    return {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "parameters": sum(p.numel() for p in model.parameters()),
        "note": "CPU-only import check; no GPU allocated",
    }


@app.function(
    image=image, cpu=4, memory=8192, timeout=900, scaledown_window=2, volumes={"/data": data_volume}
)
def prepare_data(cache: str = "cloud_full_7030"):
    import json
    import shutil

    import torch

    from photograder.cloud_data import prepare_cloud_data

    torch.set_num_threads(1)
    if not cache or Path(cache).name != cache or cache in (".", ".."):
        raise ValueError("Cache name must be a single directory name")
    destination = Path("/data") / cache
    result = prepare_cloud_data(
        ["/data/fivek/manifest.jsonl", "/data/fivek_multistyle/manifest.jsonl"],
        destination,
        destination / "split_registry.json",
        workers=4,
        expected_scenes=2748,
        expected_rows=2796,
        verification_workers=32,
    )
    terms = destination / "terms"
    terms.mkdir()
    for name in ("LicenseAdobe.txt", "LicenseAdobeMIT.txt", "SOURCE_TERMS.html"):
        shutil.copy2(Path("/data/fivek") / name, terms / name)
    for folder in ("fivek", "fivek_multistyle"):
        shutil.copy2(Path("/data") / folder / "provenance.json", terms / f"{folder}_provenance.json")
    data_volume.commit()
    summary = {
        key: result[key]
        for key in (
            "outer_scene_counts",
            "row_counts",
            "cache_bytes",
            "manifest_sha256",
            "preparation_seconds",
        )
    }
    print(json.dumps(summary), flush=True)
    return summary


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=16384,
    volumes={"/data": data_volume.with_mount_options(read_only=True), "/runs": run_volume},
    timeout=3900,
    max_containers=1,
    scaledown_window=2,
)
def run_training(
    run_id: str, cache: str = "cloud_full_7030", budget_seconds: int = 3600, resume: bool = False
):
    from photograder.cloud_runner import run

    for value in (run_id, cache):
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError("Run and cache names must be single directory names")
    if not 60 <= budget_seconds <= 3600:
        raise ValueError("Initial GPU budget must be 60..3600 seconds")
    return run(
        Path("/data") / cache,
        Path("/runs") / run_id,
        budget_seconds=budget_seconds,
        commit=run_volume.commit,
        resume=resume,
    )


@app.local_entrypoint()
def main(
    run_id: str = "full-7030-20261012",
    preflight_only: bool = False,
    prepare_only: bool = False,
    cache: str = "cloud_full_7030",
    budget_seconds: int = 3600,
    resume: bool = False,
):
    import json

    if preflight_only:
        print(json.dumps(preflight.remote(), indent=2))
        return
    if prepare_only:
        print(json.dumps(prepare_data.remote(cache), indent=2))
        return
    if not 60 <= budget_seconds <= 3600:
        raise ValueError("Initial run budget must be between 60 and 3600 GPU seconds")
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single directory name")
    result = run_training.remote(run_id, cache, budget_seconds, resume)
    destination = ROOT / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "remote_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
