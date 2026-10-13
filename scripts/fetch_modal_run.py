"""Fetch compact model artifacts and reports without duplicate optimizer states."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("Expected a single run directory name")
    destination = Path(__file__).resolve().parents[1] / "runs" / args.run_id
    volume = modal.Volume.from_name("selective-photograder-runs-v1")
    files = [
        entry.path
        for entry in volume.listdir(args.run_id, recursive=True)
        if entry.path.endswith((".json", ".csv", "/model.pt"))
    ]
    if not files:
        raise ValueError("No committed run artifacts found")

    def fetch(remote):
        relative = Path(remote.lstrip("/")).relative_to(args.run_id)
        local = destination / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_name(local.name + ".download")
        with temporary.open("wb") as stream:
            for block in volume.read_file(remote):
                stream.write(block)
        temporary.replace(local)
        return {
            "path": str(relative),
            "bytes": local.stat().st_size,
            "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, files))
    run = json.loads((destination / "run.json").read_text())
    for stage, metadata in run.get("portable_models", {}).items():
        actual = hashlib.sha256((destination / stage / "model.pt").read_bytes()).hexdigest()
        if actual != metadata["sha256"]:
            raise ValueError("Downloaded model checksum mismatch")
    (destination / "download_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": run["status"],
                "files": len(records),
                "bytes": sum(row["bytes"] for row in records),
                "destination": str(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
