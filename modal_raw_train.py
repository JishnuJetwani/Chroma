"""Public RAW acquisition on CPU, followed by bounded single-H100 experiments."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("selective-photograder-raw")
data_volume = modal.Volume.from_name("selective-photograder-data-v1")
run_volume = modal.Volume.from_name("selective-photograder-runs-v1")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements-raw.txt"), extra_options="--require-hashes")
    .apt_install("liblcms2-2", "libgomp1")
    .env(
        {
            "PYTHONPATH": "/root/project/src",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    .add_local_dir(ROOT / "src", "/root/project/src", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "data/raw_fivek_plan", "/root/plan", ignore=["official-index.html"])
    .add_local_file(
        ROOT / "configs/splits/cloud_full_7030.json", "/root/project/configs/splits/cloud_full_7030.json"
    )
)


def safe_name(value):
    if not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("Expected a single directory name")
    return value


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": data_volume},
)
def prepare(cache: str = "raw_fivek_v2", limit: int = 0):
    import hashlib
    import json
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import torch

    from photograder.raw_acquisition import download_pair
    from photograder.raw_data import RENDER_VERSION, prepare_raw_pair

    torch.set_num_threads(1)
    started = time.monotonic()
    destination = Path("/data") / safe_name(cache)
    destination.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path("/root/plan/plan.json").read_text())
    records = plan["records"][: limit or None]
    rows, errors = [], []
    (destination / "plan.json").write_text(json.dumps(plan, indent=2))
    terms = destination / "terms"
    terms.mkdir(exist_ok=True)
    for name in ("LicenseAdobe.txt", "LicenseAdobeMIT.txt", "SOURCE_TERMS.html"):
        source = Path("/root/plan") / name
        if source.exists():
            shutil.copy2(source, terms / name)
    if not any(terms.iterdir()):
        raise RuntimeError("Dataset terms missing")

    def one(record):
        sid = record["scene_id"].split(":")[-1]
        output = destination / (sid + ".npz")
        receipt = destination / (sid + ".json")
        if output.exists() and receipt.exists():
            previous = json.loads(receipt.read_text())
            if (
                previous["render_version"] == RENDER_VERSION
                and previous["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
                and previous["split"] == record["split"]
            ):
                return previous
            raise ValueError("Existing cache is inconsistent: " + sid)
        stage = Path("/tmp/raw_stage") / sid
        stage.mkdir(parents=True, exist_ok=True)
        try:
            pair = download_pair(
                record, stage, record["input_bytes"] + record["target_bytes"], min_free_bytes=2 * 1024**3
            )
            result = prepare_raw_pair(
                pair["input_path"],
                pair["target_path"],
                output,
                scene_id=record["scene_id"],
                split=record["split"],
            )
            result["source"] = {k: v for k, v in pair.items() if not k.endswith("_path")}
            receipt.write_text(json.dumps(result, indent=2))
            return result
        finally:
            # Only native files in this job's explicit staging directory are removed.
            shutil.rmtree(stage, ignore_errors=True)

    def persist():
        rows.sort(key=lambda row: (row["split"], row["scene_id"]))
        (destination / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        status = {
            "requested": len(records),
            "prepared": len(rows),
            "errors": errors,
            "counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "val")},
            "elapsed_seconds": time.monotonic() - started,
            "render_version": RENDER_VERSION,
            "cache_bytes": sum(r["bytes"] for r in rows),
        }
        (destination / "preparation.json").write_text(json.dumps(status, indent=2))
        data_volume.commit()
        print(json.dumps(status), flush=True)
        return status

    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(one, r): r for r in records}
        for future in as_completed(pending):
            record = pending[future]
            try:
                rows.append(future.result())
            except Exception as error:  # noqa: BLE001 -- report failed scenes without losing verified cache
                errors.append(
                    {
                        "scene_id": record["scene_id"],
                        "split": record["split"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            print(
                json.dumps({"completed": len(rows), "failed": len(errors), "last_scene": record["scene_id"]}),
                flush=True,
            )
            if (len(rows) + len(errors)) % 8 == 0:
                persist()
    return persist()


@app.function(
    image=image, cpu=2, memory=4096, timeout=180, scaledown_window=2, volumes={"/data": data_volume}
)
def check_cache(cache: str = "raw_fivek_v2"):
    import torch

    from photograder.raw_training import prepare_packed_cache

    torch.set_num_threads(2)
    result = prepare_packed_cache(Path("/data") / safe_name(cache))
    data_volume.commit()
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=16384,
    timeout=1800,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": data_volume.with_mount_options(read_only=True), "/runs": run_volume},
)
def train(
    run_id: str = "raw-pilot-20261030",
    cache: str = "raw_fivek_v2",
    steps: int = 800,
    budget_seconds: int = 900,
):
    from photograder.raw_training import train_pilot

    if not 20 <= steps <= 3000 or not 60 <= budget_seconds <= 1500:
        raise ValueError("Training request exceeds the bounded pilot limits")
    return train_pilot(
        Path("/data") / safe_name(cache),
        Path("/runs") / safe_name(run_id),
        steps=steps,
        budget_seconds=budget_seconds,
        commit=run_volume.commit,
    )


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=600,
    scaledown_window=2,
    volumes={"/data": data_volume.with_mount_options(read_only=True), "/runs": run_volume},
)
def verify_export(run_id: str = "raw-pilot-20261030", cache: str = "raw_fivek_v2"):
    import hashlib
    import json
    import shutil
    import time

    import numpy as np
    import tifffile
    import torch
    from PIL import Image

    from photograder.raw_acquisition import download_pair
    from photograder.raw_data import load_expert_tiff
    from photograder.raw_inference import RawGrader

    torch.set_num_threads(4)
    folder = Path("/runs") / safe_name(run_id)
    records = [
        json.loads(line)
        for line in (Path("/data") / safe_name(cache) / "manifest.jsonl").read_text().splitlines()
    ]
    row = min((r for r in records if r["split"] == "val"), key=lambda r: r["source"]["input_bytes"])
    stage = Path("/tmp/raw_export_verify")
    stage.mkdir(exist_ok=True)
    destination = folder / "native_export_check"
    destination.mkdir(exist_ok=True)
    try:
        pair = download_pair(
            row["source"],
            stage,
            row["source"]["input_bytes"] + row["source"]["target_bytes"],
            min_free_bytes=2 * 1024**3,
        )
        source = Path(pair["input_path"])
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        grader = RawGrader(folder / "shared/model.pt", device="cpu", preview_size=1024)
        analysis = grader.analyze(source)
        analysis.preview.save(destination / "before.png")
        start = time.perf_counter()
        graded = grader.render_preview(analysis)
        preview_seconds = time.perf_counter() - start
        graded.save(destination / "after.png")
        result = grader.export(analysis, stage / "result.tif")
        with tifffile.TiffFile(result["path"]) as tif:
            assert tif.pages[0].dtype == np.uint16 and 34675 in tif.pages[0].tags
            dimensions = list(tif.pages[0].shape)
        target, _ = load_expert_tiff(pair["target_path"])
        target = Image.fromarray(np.rint(target * 255).astype(np.uint8))
        target.thumbnail(graded.size)
        target.save(destination / "expert.png")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        result.update(
            {
                "scene_id": row["scene_id"],
                "analysis_seconds": analysis.analysis_seconds,
                "preview_seconds": preview_seconds,
                "tiff_shape": dimensions,
                "source_preserved": True,
                "input_max": float(analysis.linear_rgb.max()),
                "input_min": float(analysis.linear_rgb.min()),
                "device": "Modal CPU, not Mac latency",
            }
        )
        result["path"] = "Temporary native TIFF inspected and deleted after verification"
        (destination / "verification.json").write_text(json.dumps(result, indent=2))
        run_volume.commit()
        return result
    finally:
        shutil.rmtree(stage, ignore_errors=True)


@app.local_entrypoint()
def main(
    prepare_only: bool = False,
    verify_only: bool = False,
    limit: int = 0,
    cache: str = "raw_fivek_v2",
    run_id: str = "raw-pilot-20261030",
    steps: int = 800,
    budget_seconds: int = 900,
):
    import json

    if not prepare_only and not verify_only:
        print(json.dumps({"cpu_preflight": check_cache.remote(cache)}, indent=2))
    result = (
        prepare.remote(cache, limit)
        if prepare_only
        else verify_export.remote(run_id, cache)
        if verify_only
        else train.remote(run_id, cache, steps, budget_seconds)
    )
    output = ROOT / "runs" / safe_name(run_id)
    output.mkdir(parents=True, exist_ok=True)
    (
        output
        / (
            "preparation_result.json"
            if prepare_only
            else "native_export_result.json"
            if verify_only
            else "remote_result.json"
        )
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
