"""Rebuild/check frozen manifests from tracked split registries, without downloads.

Only the selected files are hashed. Existing output manifests are never changed.
Outputs must remain beside their source manifest so relative paths stay stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import read_manifest, sha256_file


def rebuild_frozen_manifest(registry_path, source_manifest, output=None):
    """Filter source rows in frozen order and verify metadata and actual bytes.

    With output=None this is a read-only check. Registry checks include exact
    canonical manifest SHA256, beyond scene/style/render/file-digest matching.
    """
    registry_path = Path(registry_path)
    source_manifest = Path(source_manifest).resolve()
    registry = json.loads(registry_path.read_text())
    if registry.get("format_version") != 1:
        raise ValueError("Unsupported frozen split registry version")
    rows = read_manifest(source_manifest)
    source_rows = {}
    for row in rows:
        key = (row["scene_id"], row["style_id"])
        if key in source_rows:
            raise ValueError(f"Ambiguous source scene/style: {key}")
        source_rows[key] = row
    scene_splits = {}
    for split, scenes in registry["splits"].items():
        if split not in {"train", "val", "test"}:
            raise ValueError("Invalid registry split")
        for scene in scenes:
            if scene in scene_splits:
                raise ValueError(f"Duplicate scene in registry splits: {scene}")
            scene_splits[scene] = split
    selected, selected_keys, digests = [], set(), {}
    for expected in registry["expected_rows"]:
        key = (expected["scene_id"], expected["style_id"])
        if key in selected_keys:
            raise ValueError(f"Duplicate expected scene/style: {key}")
        selected_keys.add(key)
        if key not in source_rows:
            raise ValueError(f"Missing frozen scene/style: {key}")
        row = source_rows[key]
        if scene_splits.get(row["scene_id"]) != row["split"]:
            raise ValueError(f"Frozen split mismatch for {key}")
        if registry["styles"].get(str(row["style_id"])) != row["style_name"]:
            raise ValueError(f"Frozen style mismatch for {key}")
        for field, value in expected.items():
            if row.get(field) != value:
                hint = (
                    " Original encoded-file hashes are required; native ICC/encoder metadata can differ even when decoded pixels agree."
                    if field == "sha256"
                    else ""
                )
                raise ValueError(f"Frozen {field} mismatch for {key}.{hint}")
        for field in ("input", "target", "valid_mask"):
            if field not in row:
                continue
            path = (source_manifest.parent / row[field]).resolve()
            if path not in digests:
                digests[path] = sha256_file(path)
            if digests[path] != expected["sha256"].get(field):
                raise ValueError(f"Actual checksum mismatch for {key}/{field}: {path}")
        selected.append(row)
    if {row["scene_id"] for row in selected} != set(scene_splits):
        raise ValueError("Registry rows and split scene inventory differ")
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected).encode()
    manifest_digest = hashlib.sha256(payload).hexdigest()
    if manifest_digest != registry["original_manifest_sha256"]:
        raise ValueError(
            "Selected manifest differs from the frozen manifest SHA256; check rendering metadata and relative paths"
        )
    if output is not None:
        output = Path(output).resolve()
        if output.parent != source_manifest.parent:
            raise ValueError("Place rebuilt manifest beside the source manifest to preserve relative paths")
        # Exclusive creation protects live training manifests, including identical ones.
        with output.open("xb") as stream:
            stream.write(payload)
    return {
        "registry": str(registry_path),
        "rows": len(selected),
        "scenes": len(scene_splits),
        "manifest_sha256": manifest_digest,
        "splits": {
            split: sum(row["split"] == split for row in selected) for split in ("train", "val", "test")
        },
        "styles": registry["styles"],
        "output": str(output) if output is not None else None,
        "actual_files_hashed": len(digests),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--source", required=True, help="Acquired source JSONL manifest")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", help="New JSONL beside source; existing files are refused")
    action.add_argument("--verify-only", action="store_true", help="Read-only exact replay check")
    args = parser.parse_args()
    print(json.dumps(rebuild_frozen_manifest(args.registry, args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
