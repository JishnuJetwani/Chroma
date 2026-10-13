"""Upload only the public, verified training cache before allocating a GPU."""

import argparse
import json
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, nargs="?")
    parser.add_argument(
        "--raw", action="store_true", help="Upload manifest-listed public pairs for CPU cloud preparation"
    )
    parser.add_argument("--volume", default="selective-photograder-data-v1")
    parser.add_argument("--destination", default="cloud_full_7030")
    args = parser.parse_args()
    volume = modal.Volume.from_name(args.volume, create_if_missing=True)
    if args.raw:
        root = Path(__file__).resolve().parents[1]
        paths = set()
        for folder in ("fivek", "fivek_multistyle"):
            manifest = root / "data" / folder / "manifest.jsonl"
            paths.add(manifest)
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                if row["source_id"] != "MIT-Adobe-FiveK":
                    raise ValueError("Only public FiveK data may be uploaded")
                for role in ("input", "target", "valid_mask"):
                    if role in row:
                        path = (manifest.parent / row[role]).resolve()
                        if not path.is_relative_to(root / "data" / "fivek") and not path.is_relative_to(
                            root / "data" / "fivek_multistyle"
                        ):
                            raise ValueError("Manifest points outside the acquired public data directories")
                        paths.add(path)
            for name in ("LicenseAdobe.txt", "LicenseAdobeMIT.txt", "SOURCE_TERMS.html", "provenance.json"):
                if (manifest.parent / name).exists():
                    paths.add(manifest.parent / name)
        print(
            f"Uploading {len(paths)} public files, {sum(path.stat().st_size for path in paths) / 1e9:.3f} GB; no GPU allocated",
            flush=True,
        )
        with volume.batch_upload(force=False) as batch:
            for path in sorted(paths):
                batch.put_file(path, "/" + str(path.relative_to(root / "data")))
        print("Public pair upload completed", flush=True)
        return
    if args.cache is None:
        parser.error("Supply cache or --raw")
    cache = args.cache.resolve()
    if not cache.is_dir() or not (cache / "index.json").is_file():
        raise ValueError("Expected a prepared cache directory containing index.json")
    # force=False prevents replacing a prior frozen dataset under the same name.
    with volume.batch_upload(force=False) as batch:
        batch.put_directory(cache, "/" + args.destination.strip("/"))
    print(f"Uploaded {cache} to {args.volume}:/{args.destination}")


if __name__ == "__main__":
    main()
