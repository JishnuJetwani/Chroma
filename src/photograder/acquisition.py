"""Bounded FiveK acquisition using HTTP ranges, never whole ZIP extraction.

Example: python -m photograder.acquisition fivek --output data/fivek_pilot
The default fetch is a 24-pair pilot. --full retrieves only the supplied official
training/test lists, deriving validation from training IDs by a stable hash.
"""

from __future__ import annotations

import argparse
import binascii
import io
import json
import shutil
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .data import sha256_file, validate_manifest, write_manifest

FIVEK_URL = "https://drive.usercontent.google.com/download?id=1Z4krjqK8a_k5eBEXmOSQc2JS_5Ioc023&export=download&confirm=t"
FIVEK_SIZE = 12252167420
GIB = 1024**3
MIN_FREE = 20 * GIB
MAX_STAGING = 20 * GIB
PREFIX = "MIT_Five_K_480p/"


class StorageGuard:
    def __init__(self, root, min_free=MIN_FREE, max_staging=MAX_STAGING):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.min_free, self.max_staging = min_free, max_staging

    def check(self, additional_bytes=0):
        used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if used + additional_bytes > self.max_staging:
            raise OSError("Dataset staging would exceed 20 GiB (or configured limit)")
        if shutil.disk_usage(self.root).free - additional_bytes < self.min_free:
            raise OSError("Download would leave less than 20 GiB free (or configured reserve)")


class RemoteZipReader(io.RawIOBase):
    """Seekable HTTP source; reject servers that ignore Range or change size."""

    def __init__(self, url, size, max_request=16 * 1024 * 1024, byte_budget=MAX_STAGING):
        self.url, self.size, self.pos = url, int(size), 0
        self.max_request, self.byte_budget, self.transferred = max_request, byte_budget, 0
        self.session = requests.Session()
        self.last_modified = None

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        position = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if position < 0 or position > self.size:
            raise ValueError("Out-of-bounds ZIP seek")
        self.pos = position
        return position

    def tell(self):
        return self.pos

    def read(self, size=-1):
        size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        result = self.range(self.pos, size)
        self.pos += len(result)
        return result

    def range(self, start, size):
        if size == 0:
            return b""
        if start < 0 or size < 0 or start + size > self.size or size > self.max_request:
            raise ValueError("Unbounded or invalid remote ZIP read")
        if self.transferred + size > self.byte_budget:
            raise OSError("Download byte budget exhausted")
        end = start + size - 1
        for attempt in range(4):
            try:
                with self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(15, 60)
                ) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise RuntimeError("Server ignored Range; refusing full archive download")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                        raise RuntimeError("Archive size/content range changed; refusing resume")
                    version = response.headers.get("Last-Modified")
                    if self.last_modified is not None and version != self.last_modified:
                        raise RuntimeError("Remote archive changed during acquisition")
                    self.last_modified = version
                    chunks, count = [], 0
                    for chunk in response.iter_content(1024 * 1024):
                        count += len(chunk)
                        self.transferred += len(chunk)
                        if count > size or self.transferred > self.byte_budget:
                            raise OSError("Server exceeded bounded range request")
                        chunks.append(chunk)
                    result = b"".join(chunks)
                    if len(result) != size:
                        raise OSError("Truncated HTTP range")
                    return result
            except (OSError, requests.RequestException):
                if attempt == 3:
                    raise
                time.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable")


def _safe_member(info):
    name = Path(info.filename)
    if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
        raise ValueError(f"Unsafe ZIP member: {info.filename}")
    if info.flag_bits & 1:
        raise ValueError("Encrypted ZIP member unsupported")
    if info.file_size > 16 * 1024 * 1024:
        raise ValueError("Member exceeds photo/text size budget")


def _member_bytes(reader, info):
    _safe_member(info)
    # Local headers can contain extra fields differing from central headers.
    head = reader.range(info.header_offset, 30)
    if head[:4] != b"PK\x03\x04":
        raise ValueError("Bad ZIP local header")
    name_len, extra_len = struct.unpack_from("<HH", head, 26)
    compressed = reader.range(info.header_offset + 30 + name_len + extra_len, info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        payload = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        payload = decoder.decompress(compressed, info.file_size + 1)
        if decoder.unconsumed_tail or not decoder.eof:
            raise ValueError("ZIP payload exceeds declared size or is truncated")
    else:
        raise ValueError("Unsupported ZIP compression")
    if len(payload) != info.file_size or binascii.crc32(payload) & 0xFFFFFFFF != info.CRC:
        raise ValueError("ZIP checksum/size mismatch")
    return payload


def _store_member(reader, info, destination):
    destination = Path(destination)
    if destination.exists():
        payload = destination.read_bytes()
        if len(payload) == info.file_size and binascii.crc32(payload) & 0xFFFFFFFF == info.CRC:
            return False
        raise ValueError(f"Existing downloaded member is corrupt: {destination}")
    payload = _member_bytes(reader, info)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return True


def _retain_fivek_terms(root):
    """Retain original research license texts and file applicability lists."""
    for name in ("LicenseAdobe.txt", "LicenseAdobeMIT.txt", "filesAdobe.txt", "filesAdobeMIT.txt"):
        path = Path(root) / name
        if path.is_file():
            continue
        url = "https://data.csail.mit.edu/graphics/fivek/legal/" + name
        with requests.get(url, timeout=(15, 30), stream=True) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(8192):
                content.extend(chunk)
                if len(content) > 1024 * 1024:
                    raise ValueError("Unexpected oversized dataset license response")
        path.write_bytes(bytes(content))


def acquire_fivek(output, full=False, train_count=16, val_count=4, test_count=4, workers=4):
    """Fetch original archive members with CRC verification and per-file resume.

    Published test IDs never move into training. Validation is the lowest 10%
    of SHA256('fivek-val-v1:'+ID) values from the supplied training list. This
    mirror's missing train_label.txt is documented, never guessed from leftovers.
    """
    import hashlib

    if not 1 <= workers <= 32:
        raise ValueError("workers must be in1..32")
    if min(train_count, val_count, test_count) < 1:
        raise ValueError("Pilot split counts must be positive")
    root = Path(output).resolve()
    guard = StorageGuard(root)
    guard.check()
    _retain_fivek_terms(root)
    reader = RemoteZipReader(FIVEK_URL, FIVEK_SIZE, byte_budget=MAX_STAGING if full else 100 * 1024 * 1024)
    archive = zipfile.ZipFile(reader)
    info = {i.filename: i for i in archive.infolist()}
    split_lists = {}
    for name in ("train_input.txt", "test.txt"):
        _store_member(reader, info[PREFIX + name], root / name)
        split_lists[name] = (root / name).read_text().split()
    train_ids = sorted(set(split_lists["train_input.txt"]))
    test_ids = sorted(set(split_lists["test.txt"]))
    if set(train_ids) & set(test_ids):
        raise ValueError("Published source lists leak test into training")
    val_ids = set(
        sorted(train_ids, key=lambda s: hashlib.sha256(("fivek-val-v1:" + s).encode()).hexdigest())[
            : max(1, len(train_ids) // 10)
        ]
    )
    splits = {"train": [i for i in train_ids if i not in val_ids], "val": sorted(val_ids), "test": test_ids}
    if not full:
        for split, count in (("train", train_count), ("val", val_count), ("test", test_count)):
            # Stable spread through the source list instead of only its first scenes.
            candidates = splits[split]
            splits[split] = [
                candidates[int(i * len(candidates) / count)] for i in range(min(count, len(candidates)))
            ]
    jobs = []
    for _split, ids in splits.items():
        for photo_id in ids:
            for folder in ("input", "expertC"):
                member = f"{PREFIX}{folder}/JPG/480p/{photo_id}.jpg"
                if member not in info:
                    raise ValueError(f"Published paired photo missing from archive: {member}")
                jobs.append((info[member], root / folder / f"{photo_id}.jpg"))
    total = sum(i.file_size for i, _ in jobs)
    if not full and total > 95 * 1024 * 1024:
        raise OSError("Pilot would exceed 100 MiB acquisition limit")
    guard.check(sum(i.file_size for i, path in jobs if not path.exists()))
    print(
        json.dumps(
            {
                "status": "acquiring",
                "pairs": sum(map(len, splits.values())),
                "uncompressed_bytes": total,
                "splits": {k: len(v) for k, v in splits.items()},
            }
        ),
        flush=True,
    )
    jobs.sort(key=lambda job: job[0].header_offset)

    def download(job):
        remote = RemoteZipReader(FIVEK_URL, FIVEK_SIZE, byte_budget=32 * 1024 * 1024)
        remote.last_modified = reader.last_modified
        item, path = job
        _store_member(remote, item, path)
        return remote.transferred

    transferred = reader.transferred
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, size in enumerate(pool.map(download, jobs), 1):
            transferred += size
            if completed % 100 == 0 or completed == len(jobs):
                guard.check()
                print(
                    json.dumps(
                        {"files_complete": completed, "files_total": len(jobs), "network_bytes": transferred}
                    ),
                    flush=True,
                )
    rows = []
    for split, ids in splits.items():
        for photo_id in ids:
            row = {
                "source_id": "MIT-Adobe-FiveK",
                "scene_id": "fivek:" + photo_id,
                "input": f"input/{photo_id}.jpg",
                "target": f"expertC/{photo_id}.jpg",
                "style_id": 0,
                "style_name": "FiveK Expert C",
                "split": split,
                "render_version": "HuiZeng-480p-srgb-jpeg-exif-v1",
            }
            row["sha256"] = {key: sha256_file(root / row[key]) for key in ("input", "target")}
            rows.append(row)
    manifest = root / "manifest.jsonl"
    write_manifest(rows, manifest)
    provenance = {
        "dataset": "MIT-Adobe FiveK",
        "source": "https://data.csail.mit.edu/graphics/fivek/",
        "mirror_documentation": "https://github.com/HuiZeng/Image-Adaptive-3DLUT",
        "mirror_url": FIVEK_URL,
        "archive_bytes": FIVEK_SIZE,
        "archive_last_modified": reader.last_modified,
        "archive_sha256": None,
        "member_integrity": "ZIP CRC32 checked; SHA256 of each downloaded file recorded",
        "acquisition": "HTTP Range: original JPG/480p members only; no full archive downloaded",
        "network_bytes_this_run": transferred,
        "pilot": not full,
        "terms_url": "https://data.csail.mit.edu/graphics/fivek/",
        "usage": "Research/non-commercial only; original dataset terms apply. No dataset redistribution in source repository.",
        "rendering": "Prepared paired sRGB JPEG release by Image-Adaptive-3DLUT; embedded ICC honored at load; missing profiles assumed sRGB. Archive original RAW/PNG representations not fetched.",
        "split_policy": "Published test.txt retained. Validation is lowest10% SHA256('fivek-val-v1:'+ID) of train_input.txt. Archive omits train_label.txt; missing IDs excluded, not inferred.",
        "split_file_sha256": {name: sha256_file(root / name) for name in split_lists},
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(validate_manifest(manifest), indent=2), flush=True)
    return manifest


def _verify_native_icc_precision(icc, srgb_profile, stage):
    """Prove ColorSync responds to the low8bits of16-bit source samples.

    Each row has identical high bytes, so prematurely quantizing the input makes
    every row constant. Native output variation demonstrates low-byte use before
    output quantization. This checks the running ColorSync implementation.
    """
    import subprocess
    import tempfile

    import numpy as np
    from PIL import Image

    from .data import load_srgb

    def chunk(kind, payload):
        block = kind + payload
        return struct.pack(">I", len(payload)) + block + struct.pack(">I", zlib.crc32(block))

    rng = np.random.default_rng(3)
    pixels = np.zeros((64, 256, 3), dtype=np.uint16)
    for row in range(64):
        pixels[row] = rng.integers(20, 190, size=(1, 3), dtype=np.uint16) * 256
        pixels[row, :, 0] += np.arange(256, dtype=np.uint16)
    scanlines = b"".join(b"\x00" + row.astype(">u2").tobytes() for row in pixels)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 64, 16, 2, 0, 0, 0))
        + chunk(b"iCCP", b"Source ICC\x00\x00" + zlib.compress(icc))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    with tempfile.TemporaryDirectory(dir=stage) as directory:
        source = Path(directory) / "precision16.png"
        output = Path(directory) / "native_srgb.png"
        source.write_bytes(payload)
        subprocess.run(
            [
                "/usr/bin/sips",
                "-m",
                str(srgb_profile),
                "-s",
                "format",
                "png",
                str(source),
                "--out",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        native = np.asarray(Image.open(output).convert("RGB"), dtype=np.int16)
        quantized_first = np.asarray(load_srgb(source), dtype=np.int16)
        native_varied = int((np.ptp(native, axis=1) > 0).any(axis=1).sum())
        quantized_varied = int((np.ptp(quantized_first, axis=1) > 0).any(axis=1).sum())
    if not native_varied or quantized_varied:
        raise RuntimeError(
            "Cannot verify native16-bit ICC transform precision; refusing early8-bit conversion"
        )
    return {
        "probe": "constant-high-byte varying-low-byte16-bit RGB ramp",
        "rows": 64,
        "native_rows_with_variation": native_varied,
        "prequantized_rows_with_variation": quantized_varied,
        "status": "Native transform verified to use low8sourcebits before outputquantization",
    }


def acquire_fivek_experts(
    manifest, output, train_count=16, val_count=4, test_count=4, max_network_bytes=3 * GIB
):
    """Native-ICC A/B fallback for a bounded subset of an existing FiveK split.

    macOS ColorSync processes16-bit ProPhoto source pixels before8-bit sRGB
    output. Full-resolution TIFFs are staged individually, transformed with
    ``sips``, resized to the exact existing input dimensions, then deleted.
    Geometry matching does not establish pixel-perfect registration; visually
    audit the selected pairs before interpreting quantitative style results.
    """
    import os
    import re
    import subprocess

    from PIL import Image

    from .data import load_srgb, read_manifest

    source_manifest = Path(manifest).resolve()
    root = Path(output).resolve()
    guard = StorageGuard(root)
    guard.check()
    profile = Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc")
    if not Path("/usr/bin/sips").exists() or not profile.exists():
        raise RuntimeError(
            "Additional-expert import requires macOS sips/ColorSync; no quantized PIL fallback"
        )
    if min(train_count, val_count, test_count) < 1:
        raise ValueError("All subset split counts must be positive")
    rows = read_manifest(source_manifest)
    candidates = [
        r for r in rows if r["source_id"] == "MIT-Adobe-FiveK" and r["style_name"] == "FiveK Expert C"
    ]
    selected = []
    for split, count in (("train", train_count), ("val", val_count), ("test", test_count)):
        pool = sorted((r for r in candidates if r["split"] == split), key=lambda r: r["scene_id"])
        if len(pool) < count:
            raise ValueError(f"Not enough existing {split} scenes for the requested subset")
        selected.extend(pool[:count])
    _retain_fivek_terms(root)
    base_url = "https://data.csail.mit.edu/graphics/fivek/"
    with requests.get(base_url, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        content = bytearray()
        for block in response.iter_content(1024 * 1024):
            content.extend(block)
            if len(content) > 8 * 1024 * 1024:
                raise ValueError("Unexpected oversized FiveK index")
    index_html = bytes(content).decode("utf-8")
    network_used = len(content)
    planned = []
    for row in selected:
        photo_id = row["scene_id"].split(":", 1)[1]
        for letter, style_id in (("a", 1), ("b", 2)):
            pattern = r'href="(img/tiff16_' + letter + "/" + re.escape(photo_id) + r'-[^"/]+\.tif)"'
            matches = re.findall(pattern, index_html)
            if len(matches) != 1:
                raise ValueError(f"No unique official TIFF URL for {photo_id}/{letter}")
            url = base_url + matches[0]
            response = requests.head(url, timeout=(15, 30))
            response.raise_for_status()
            size = int(response.headers.get("Content-Length", "0"))
            if size <= 0 or size > 160 * 1024 * 1024:
                raise ValueError(f"Unexpected TIFF size for {photo_id}: {size}")
            planned.append((row, letter, style_id, url, size))
    if network_used + sum(item[-1] for item in planned) > max_network_bytes:
        raise OSError(
            "Additional-expert source TIFFs would exceed configured network budget; reduce subset counts"
        )
    print(
        json.dumps(
            {
                "status": "acquiring_experts",
                "scenes": len(selected),
                "source_tiffs": len(planned),
                "planned_network_bytes": network_used + sum(item[-1] for item in planned),
            }
        ),
        flush=True,
    )
    stage = root / ".staging"
    stage.mkdir(exist_ok=True)
    imported = []
    for row in selected:
        converted = dict(row)
        for key in ("input", "target", "valid_mask"):
            if key in row:
                converted[key] = os.path.relpath(source_manifest.parent / row[key], root)
        imported.append(converted)
    source_records = []
    precision_report = None
    for number, (row, letter, style_id, url, size) in enumerate(planned, 1):
        photo_id = row["scene_id"].split(":", 1)[1]
        filename = f"{photo_id}_expert{letter.upper()}"
        target = root / (filename + ".png")
        metadata_file = root / (filename + ".source.json")
        input_path = source_manifest.parent / row["input"]
        expected_size = load_srgb(input_path).size
        if target.exists() and metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            if (
                metadata["url"] != url
                or metadata["source_bytes"] != size
                or metadata["output_sha256"] != sha256_file(target)
            ):
                raise ValueError(f"Expert resume metadata/checksum mismatch: {target}")
        else:
            guard.check(size * 4)
            original = stage / (filename + ".tif")
            native = stage / (filename + ".srgb.tif")
            temporary = original.with_suffix(".partial")
            try:
                with requests.get(url, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    received = 0
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            received += len(block)
                            network_used += len(block)
                            if received > size or network_used > max_network_bytes:
                                raise OSError("Expert download exceeded declared size/network budget")
                            stream.write(block)
                    if received != size:
                        raise OSError("Truncated expert TIFF download")
                temporary.replace(original)
                original_sha256 = sha256_file(original)
                with Image.open(original) as source:
                    bits = tuple(source.tag_v2.get(258, ()))
                    source_size = source.size
                    icc = source.info.get("icc_profile")
                    if bits != (16, 16, 16) or not icc:
                        raise ValueError("Expected16-bit RGB TIFF with original embedded ICC profile")
                    orientation = source.getexif().get(274, 1)
                    if orientation in (5, 6, 7, 8):
                        source_size = source_size[::-1]
                if precision_report is None:
                    precision_report = _verify_native_icc_precision(icc, profile, stage)
                sw, sh = source_size
                tw, th = expected_size
                if abs(sw * th / sh - tw) > 1 or abs(sh * tw / sw - th) > 1:
                    raise ValueError(
                        f"Expert aspect ratio differs beyond rounding: {photo_id}/{letter}, {source_size} vs {expected_size}"
                    )
                subprocess.run(
                    [
                        "/usr/bin/sips",
                        "-m",
                        str(profile),
                        "-s",
                        "format",
                        "tiff",
                        str(original),
                        "--out",
                        str(native),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                with Image.open(native) as transformed:
                    native_bits = tuple(transformed.tag_v2.get(258, ()))
                    native_profile = transformed.info.get("icc_profile")
                    if native_bits != (8, 8, 8) or not native_profile:
                        raise ValueError("Unexpected ColorSync output bit depth/profile")
                # Native ICC transform already happened before8-bit decode here.
                image = load_srgb(native).resize(expected_size, Image.Resampling.LANCZOS)
                image.save(target)
                metadata = {
                    "url": url,
                    "source_bytes": size,
                    "source_sha256": original_sha256,
                    "source_dimensions": list(source_size),
                    "source_bits": list(bits),
                    "output_dimensions": list(expected_size),
                    "output_sha256": sha256_file(target),
                    "native_color_engine": "macOS sips/ColorSync",
                    "native_profile_sha256": sha256_file(profile),
                    "native_output_bits": list(native_bits),
                    "native_precision_probe": precision_report,
                    "render_version": "fivek-colorsync16input-srgb8-lanczos-v1",
                }
                metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
            finally:
                for file in (original, native, temporary):
                    file.unlink(missing_ok=True)
        converted = dict(row)
        converted.update(
            style_id=style_id,
            style_name=f"FiveK Expert {letter.upper()}",
            input=os.path.relpath(input_path, root),
            target=target.name,
            render_version="fivek-colorsync16input-srgb8-lanczos-v1",
            source_target_sha256=metadata["source_sha256"],
        )
        converted["sha256"] = {"input": sha256_file(input_path), "target": sha256_file(target)}
        imported.append(converted)
        source_records.append(metadata)
        print(
            json.dumps(
                {
                    "expert_targets_complete": number,
                    "total": len(planned),
                    "network_bytes_this_run": network_used,
                }
            ),
            flush=True,
        )
    output_manifest = root / "manifest.jsonl"
    write_manifest(imported, output_manifest)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "dataset": "MIT-Adobe FiveK additional experts",
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": sha256_file(source_manifest),
                "index_url": base_url,
                "network_bytes_this_run": network_used,
                "network_budget_bytes": max_network_bytes,
                "rendering": "Official16-bit ProPhotoRGB TIFF->macOS ColorSync nativeICC transform->8-bit sRGB TIFF->LANCZOS resize to existinginput->PNG",
                "precision_validation": precision_report
                or next(
                    (m["native_precision_probe"] for m in source_records if "native_precision_probe" in m),
                    {"status": "Previously imported outputs; precision probe not rerun"},
                ),
                "alignment": "Source aspect ratio within1pixel at outputsize; exact final dimensions. This is not subpixelregistration validation; visually audit real paired images.",
                "splits": "Exact existing source scene splits; no new scene reassignment.",
                "terms": "Original research/non-commercial licenses retained alongside the data.",
                "staging": "One sourceTIFF andnativeconvertedTIFF at a time; both deleted after sourcehash/provenance/output persisted.",
                "source_records": source_records,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(validate_manifest(output_manifest), indent=2), flush=True)
    return output_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fivek = sub.add_parser("fivek")
    fivek.add_argument("--output", default="data/fivek_pilot")
    fivek.add_argument("--full", action="store_true")
    fivek.add_argument("--train-count", type=int, default=16)
    fivek.add_argument("--val-count", type=int, default=4)
    fivek.add_argument("--test-count", type=int, default=4)
    fivek.add_argument("--workers", type=int, default=4)
    s24 = sub.add_parser("s24", help="Import a locally supplied official S24 directory")
    s24.add_argument("--source", required=True)
    s24.add_argument("--output", required=True)
    s24.add_argument("--styles", default="0,1,2")
    s24.add_argument("--limit-per-split", type=int)
    s24.add_argument("--raw-orientation", choices=("metadata", "already_oriented"), default="metadata")
    experts = sub.add_parser(
        "fivek-experts", help="Bounded nativeColorSync import of officialFiveK A/B TIFFs"
    )
    experts.add_argument("--manifest", required=True)
    experts.add_argument("--output", default="data/fivek_multistyle")
    experts.add_argument("--train-count", type=int, default=16)
    experts.add_argument("--val-count", type=int, default=4)
    experts.add_argument("--test-count", type=int, default=4)
    experts.add_argument("--network-gib", type=float, default=3)
    args = parser.parse_args()
    if args.command == "fivek":
        acquire_fivek(args.output, args.full, args.train_count, args.val_count, args.test_count, args.workers)
    elif args.command == "fivek-experts":
        acquire_fivek_experts(
            args.manifest,
            args.output,
            args.train_count,
            args.val_count,
            args.test_count,
            int(args.network_gib * GIB),
        )
    else:
        from .data import import_s24

        manifest = import_s24(
            args.source,
            args.output,
            tuple(map(int, args.styles.split(","))),
            args.limit_per_split,
            args.raw_orientation,
        )
        print(json.dumps(validate_manifest(manifest), indent=2))


if __name__ == "__main__":
    main()
