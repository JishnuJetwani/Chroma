import copy
import hashlib
import json

import pytest
from PIL import Image

from photograder.data import sha256_file, write_manifest
from photograder.splits import rebuild_frozen_manifest


def frozen_fixture(tmp_path):
    rows = []
    for number, split in enumerate(("train", "train", "val")):
        paths = {}
        for role, offset in (("input", 0), ("target", 9)):
            path = tmp_path / f"{number}_{role}.png"
            Image.new("RGB", (16, 16), (number * 40 + offset, 90, 120)).save(path)
            paths[role] = path.name
        row = {
            "source_id": "fixture",
            "scene_id": f"fixture:{number}",
            "split": split,
            "style_id": 0,
            "style_name": "Fixture style",
            "render_version": "fixture-v1",
            **paths,
        }
        row["sha256"] = {key: sha256_file(tmp_path / name) for key, name in paths.items()}
        rows.append(row)
    full = tmp_path / "full.jsonl"
    write_manifest(rows, full)
    selected = rows[1:]
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    registry = {
        "format_version": 1,
        "styles": {"0": "Fixture style"},
        "splits": {"train": ["fixture:1"], "val": ["fixture:2"], "test": []},
        "original_manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "expected_rows": [
            {
                key: row[key]
                for key in ("source_id", "scene_id", "split", "style_id", "render_version", "sha256")
            }
            for row in selected
        ],
    }
    frozen = tmp_path / "registry.json"
    frozen.write_text(json.dumps(registry))
    return full, frozen, rows, payload


def test_rebuild_exact_frozen_bytes_without_changing_source(tmp_path):
    source, registry, rows, payload = frozen_fixture(tmp_path)
    before = source.read_bytes()
    output = tmp_path / "preview.jsonl"
    result = rebuild_frozen_manifest(registry, source, output)
    assert output.read_text() == payload
    assert result["rows"] == 2 and result["scenes"] == 2
    assert source.read_bytes() == before
    result_readonly = rebuild_frozen_manifest(registry, source)
    assert result_readonly["output"] is None
    assert result_readonly["manifest_sha256"] == sha256_file(output)
    with pytest.raises(FileExistsError):
        rebuild_frozen_manifest(registry, source, output)
    assert output.read_text() == payload


@pytest.mark.parametrize(
    "change,error",
    [
        ("missing", "Missing frozen"),
        ("split", "split mismatch"),
        ("style", "style mismatch"),
        ("digest", "sha256 mismatch"),
    ],
)
def test_rebuild_rejects_source_drift(tmp_path, change, error):
    source, registry, rows, _ = frozen_fixture(tmp_path)
    altered = copy.deepcopy(rows)
    if change == "missing":
        altered.pop(1)
    elif change == "split":
        altered[1]["split"] = "val"
    elif change == "style":
        for row in altered:
            row["style_name"] = "Different style"
    else:
        altered[1]["sha256"]["target"] = "0" * 64
    changed_source = tmp_path / "changed.jsonl"
    write_manifest(altered, changed_source)
    output = tmp_path / "must_not_exist.jsonl"
    with pytest.raises(ValueError, match=error):
        rebuild_frozen_manifest(registry, changed_source, output)
    assert not output.exists()


def test_rebuild_verifies_actual_image_bytes_and_full_manifest_hash(tmp_path):
    source, registry, rows, _ = frozen_fixture(tmp_path)
    target = tmp_path / rows[1]["target"]
    original = target.read_bytes()
    target.write_bytes(b"changed image content")
    with pytest.raises(ValueError, match="Actual checksum"):
        rebuild_frozen_manifest(registry, source)
    target.write_bytes(original)
    frozen = json.loads(registry.read_text())
    frozen["original_manifest_sha256"] = "0" * 64
    registry.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="frozen manifest SHA256"):
        rebuild_frozen_manifest(registry, source)
