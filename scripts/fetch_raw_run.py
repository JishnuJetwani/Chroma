"""Download RAW research checkpoints/previews and verify their recorded hashes."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", nargs="?", default="raw-pilot-20261030")
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("Run ID must be a single directory name")
    root = Path(__file__).resolve().parents[1] / "runs" / args.run_id
    volume = modal.Volume.from_name("selective-photograder-runs-v1")
    paths = [
        entry.path
        for entry in volume.listdir(args.run_id, recursive=True)
        if entry.path.endswith((".json", ".png", ".py", "/model.pt"))
    ]
    if not paths:
        raise ValueError("No committed RAW artifacts found")

    def fetch(remote):
        relative = Path(remote.lstrip("/")).relative_to(args.run_id)
        if ".." in relative.parts:
            raise ValueError("Invalid remote artifact path")
        local = root / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_suffix(local.suffix + ".download")
        digest = hashlib.sha256()
        with temporary.open("wb") as stream:
            for block in volume.read_file(remote):
                stream.write(block)
                digest.update(block)
        temporary.replace(local)
        return {"path": str(relative), "sha256": digest.hexdigest(), "bytes": local.stat().st_size}

    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(fetch, paths))
    report = json.loads((root / "report.json").read_text())
    if not report["complete"]:
        raise ValueError("RAW experiment is incomplete; do not promote it")
    hashes = {row["path"]: row["sha256"] for row in downloaded}
    for arm, result in report["arms"].items():
        if hashes.get(arm + "/model.pt") != result["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch: " + arm)
    (root / "download_manifest.json").write_text(json.dumps(downloaded, indent=2) + "\n")
    print(
        json.dumps(
            {
                "files": len(downloaded),
                "bytes": sum(r["bytes"] for r in downloaded),
                "destination": str(root),
                "checkpoints_verified": list(report["arms"]),
            }
        )
    )


if __name__ == "__main__":
    main()
