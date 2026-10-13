"""Launch Modal with a separate process session and durable local logs."""

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--cache", default="cloud_full_7030_io32")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for value in (args.run_id, args.cache):
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError("Run and cache must be single directory names")
    root = Path(__file__).resolve().parents[1]
    destination = root / "runs" / args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    previous = destination / "launcher.json"
    if previous.exists():
        record = json.loads(previous.read_text())
        try:
            os.kill(record["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Recorded launcher process still exists; inspect it before starting another")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    logfile = destination / f"modal-{stamp}.log"
    command = [
        str(root / ".venv/bin/modal"),
        "run",
        "--detach",
        "modal_train.py",
        "--run-id",
        args.run_id,
        "--cache",
        args.cache,
    ]
    if args.resume:
        command.append("--resume")
    with logfile.open("xb") as stream:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {
        "pid": process.pid,
        "started_at_utc": stamp,
        "log": str(logfile),
        "command": command,
        "resume": args.resume,
    }
    previous.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
