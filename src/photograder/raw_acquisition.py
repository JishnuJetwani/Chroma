"""Bounded acquisition of official FiveK sensor DNGs and Expert C TIFFs.

This module downloads sources only. Color conversion, geometry validation and
prepared-cache creation belong to the RAW preparation layer. No source images
are fetched by ``plan``. Defaults preserve the existing 20 GiB disk reserve.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

from .acquisition import MIN_FREE, StorageGuard, _retain_fivek_terms
from .data import sha256_file

BASE_URL = "https://data.csail.mit.edu/graphics/fivek/"
INDEX_LIMIT = 8 * 1024**2
FILE_LIMIT = 192 * 1024**2
INNER_VAL_COUNT = 192
PARTITION_VERSION = "raw-fivek-inner-val-v1"
SELECTION_VERSION = "raw-fivek-pilot-v1"


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def dng_crop_metadata(path_or_buffer, *, require_integer_crop=True):
    """Read DNG crop tags from the largest CFA IFD, including nested SubIFDs.

    No pixel payload is decoded. Rational tags are converted using their actual
    denominators. Crop coordinates are relative to ActiveArea, whereas RawPy's
    visible array may already exclude sensor margins. Apply that difference in
    the caller before cropping, and apply EXIF orientation after the crop.
    Noninteger crops raise by default instead of rounding to a different image.
    """
    import tifffile

    source = io.BytesIO(path_or_buffer) if isinstance(path_or_buffer, (bytes, bytearray)) else path_or_buffer

    def values(tag, count, name):
        if tag is None:
            return None
        value = tag.value
        value = list(value) if isinstance(value, (tuple, list)) else [value]
        if int(tag.dtype) in (5, 10):  # TIFF RATIONAL/SRATIONAL have flat numerator/denominator pairs.
            if len(value) != 2 * count or any(float(value[2 * i + 1]) == 0 for i in range(count)):
                raise ValueError(f"Invalid rational DNG {name}")
            value = [float(value[2 * i]) / float(value[2 * i + 1]) for i in range(count)]
        else:
            value = [float(x) for x in value]
        if len(value) != count or not all(math.isfinite(x) for x in value):
            raise ValueError(f"Invalid DNG {name}")
        return [int(x) if float(x).is_integer() else x for x in value]

    with tifffile.TiffFile(source) as document:
        first = document.pages[0]
        if first.tags.get(50706) is None:
            raise ValueError("Input TIFF does not identify itself as DNG")
        candidates = []
        visited = set()

        def walk(pages, prefix=(), depth=0):
            if depth > 8:
                raise ValueError("Unexpectedly deep DNG SubIFDs")
            for i, page in enumerate(pages):
                if page.offset in visited or len(visited) >= 64:
                    raise ValueError("Repeated or excessive DNG IFDs")
                visited.add(page.offset)
                path = (*prefix, i)
                if int(page.photometric) == 32803:
                    candidates.append((int(page.imagewidth) * int(page.imagelength), page, path))
                if page.subifds:
                    walk(page.pages, path, depth + 1)

        walk(document.pages)
        if not candidates:
            raise ValueError("No CFA sensor IFD found in DNG; do not use its RGB thumbnail")
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            raise ValueError("Ambiguous equally sized CFA sensor IFDs")
        _, sensor, ifd_path = candidates[0]

        def get_tag(code):
            # DNG crop tags normally belong to the raw SubIFD. A root tag is a
            # legitimate fallback for DNGs whose raw metadata is in IFD0.
            return sensor.tags.get(code) or first.tags.get(code)

        width, height = int(sensor.imagewidth), int(sensor.imagelength)
        active = values(get_tag(50829), 4, "ActiveArea")
        active_defaulted = active is None
        active = [0, 0, height, width] if active is None else active
        if (
            any(not isinstance(x, int) for x in active)
            or not 0 <= active[0] < active[2] <= height
            or not 0 <= active[1] < active[3] <= width
        ):
            raise ValueError("DNG ActiveArea is not an integer sensor rectangle")
        origin = values(get_tag(50719), 2, "DefaultCropOrigin")
        crop_size = values(get_tag(50720), 2, "DefaultCropSize")
        scale = values(get_tag(50718), 2, "DefaultScale") or [1, 1]
        if (origin is None) != (crop_size is None):
            raise ValueError("DNG contains incomplete default crop metadata")
        if origin is not None:
            if require_integer_crop and any(not isinstance(x, int) for x in origin + crop_size):
                raise ValueError(f"Noninteger DNG crop requires explicit resampling: {origin}, {crop_size}")
            if (
                min(origin) < 0
                or min(crop_size) <= 0
                or origin[0] + crop_size[0] > active[3] - active[1]
                or origin[1] + crop_size[1] > active[2] - active[0]
            ):
                raise ValueError("DNG default crop lies outside ActiveArea")
        orientation_tag = get_tag(274)
        orientation = int(orientation_tag.value) if orientation_tag is not None else 1
        if orientation not in range(1, 9) or min(scale) <= 0:
            raise ValueError("Invalid DNG orientation or default scale")
        return {
            "active_area_yxyx": active,
            "active_area_defaulted": active_defaulted,
            "default_crop_origin_xy": origin,
            "default_crop_size_wh": crop_size,
            "default_scale_xy": scale,
            "exif_orientation": orientation,
            "fullsensor_size_wh": [width, height],
            "ifd_index_path": list(ifd_path),
            "ifd_offset": int(sensor.offset),
            "has_default_crop": origin is not None,
            "crop_coordinates": "relative-to-active-area-before-exif-orientation",
        }


def fetch_official_index(output=None):
    """Fetch at most 8 MiB of official HTML; optionally retain its SHA256."""
    with requests.get(BASE_URL, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        content = bytearray()
        for block in response.iter_content(64 * 1024):
            content.extend(block)
            if len(content) > INDEX_LIMIT:
                raise ValueError("Official FiveK index exceeds bounded HTML budget")
    payload = bytes(content)
    index_html = payload.decode("utf-8")
    # Parse before accepting an index as usable acquisition metadata.
    parse_official_index(index_html)
    if output is not None:
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        (root / "official-index.html").write_bytes(payload)
        _atomic_json(
            root / "official-index.source.json",
            {
                "url": BASE_URL,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        )
    return index_html


def parse_official_index(index_html):
    """Map canonical FiveK scene IDs to exact URLs linked by the author."""
    if len(index_html.encode("utf-8")) > INDEX_LIMIT:
        raise ValueError("Official FiveK index exceeds bounded HTML budget")
    result = {}
    for table_row in re.findall(r"<tr>(.*?)</tr>", index_html, flags=re.DOTALL):
        inputs = re.findall(r'href="(img/dng/(a\d{4})-[^"/]+\.dng)"', table_row)
        if not inputs:
            continue
        if len(inputs) != 1:
            raise ValueError("Ambiguous official DNG link")
        input_path, photo_id = inputs[0]
        targets = re.findall(r'href="(img/tiff16_c/' + photo_id + r'-[^"/]+\.tif)"', table_row)
        if len(targets) != 1 or Path(input_path).stem != Path(targets[0]).stem:
            raise ValueError(f"Missing or mismatched official Expert C pair: {photo_id}")
        scene_id = "fivek:" + photo_id
        if scene_id in result:
            raise ValueError(f"Duplicate official scene: {scene_id}")
        exif = re.search(r"EXIF:<br>(.*?)</td>", table_row, flags=re.DOTALL)
        # The index supplies a human-readable camera/exposure description, not
        # a separate machine-readable camera model. Keep its exact provenance.
        result[scene_id] = {
            "scene_id": scene_id,
            "source_id": "MIT-Adobe-FiveK",
            "input_url": BASE_URL + html.unescape(input_path),
            "target_url": BASE_URL + html.unescape(targets[0]),
            "camera_exif_description": html.unescape(exif.group(1).strip()) if exif else None,
            "input_encoding": "original-sensor-dng",
            "target_encoding": "16bit-prophoto-rgb-tiff",
            "style_id": 0,
            "style_name": "FiveK Expert C",
        }
    if len(result) != 5000:
        raise ValueError(f"Expected 5,000 official RAW/Expert C pairs, found {len(result)}")
    return result


def _rank(scene_id, namespace):
    return hashlib.sha256((namespace + ":" + scene_id).encode()).hexdigest()


def plan_fivek_raw(registry, index_html, train_count=128, val_count=32):
    """Select nested pilot subsets while keeping all outer-test scenes unused.

    Inner validation is fixed to 192 of the 1,924 frozen outer-training scenes.
    Changing pilot sizes cannot move any scene between train and validation.
    No outer-test acquisition is implemented in this pilot function.
    """
    registry_path = Path(registry)
    registry_data = json.loads(registry_path.read_text())
    train = registry_data["outer_splits"]["train"]
    test = registry_data["outer_splits"]["test"]
    if len(train) != 1924 or len(set(train)) != 1924 or len(test) != 824 or len(set(test)) != 824:
        raise ValueError("Expected the frozen 1,924/824 scene registry")
    if set(train) & set(test):
        raise ValueError("Frozen outer split contains scene leakage")
    if not 1 <= train_count <= len(train) - INNER_VAL_COUNT or not 1 <= val_count <= INNER_VAL_COUNT:
        raise ValueError("Requested pilot counts exceed the fixed inner partitions")
    validation = set(sorted(train, key=lambda sid: _rank(sid, PARTITION_VERSION))[:INNER_VAL_COUNT])
    pools = {"train": set(train) - validation, "val": validation}
    index = parse_official_index(index_html)
    records = []
    for split, count in (("train", train_count), ("val", val_count)):
        selected = sorted(pools[split], key=lambda sid: _rank(sid, SELECTION_VERSION))[:count]
        for scene_id in selected:
            if scene_id not in index:
                raise ValueError(f"Frozen scene absent from official source index: {scene_id}")
            records.append(dict(index[scene_id], split=split, outer_split="train"))
    return {
        "format_version": 1,
        "dataset": "MIT-Adobe FiveK original RAW / Expert C",
        "index_url": BASE_URL,
        "index_sha256": hashlib.sha256(index_html.encode("utf-8")).hexdigest(),
        "registry_sha256": sha256_file(registry_path),
        "partition_version": PARTITION_VERSION,
        "selection_version": SELECTION_VERSION,
        "inner_partition_sizes": {"train": 1732, "val": INNER_VAL_COUNT},
        "selected_counts": {"train": train_count, "val": val_count},
        "outer_test_used": False,
        "source_sha256_available_before_download": False,
        "records": records,
    }


def _validate_source_url(url, kind):
    prefix = BASE_URL + ("img/dng/" if kind == "input" else "img/tiff16_c/")
    parsed = urlparse(url)
    if not url.startswith(prefix) or parsed.query or parsed.fragment or ".." in parsed.path:
        raise ValueError("Only exact official FiveK source URLs are accepted")


def probe_pair_sizes(record):
    """Use HEAD requests only and add exact input_bytes/target_bytes."""
    result = dict(record)
    for kind in ("input", "target"):
        url = record[kind + "_url"]
        _validate_source_url(url, kind)
        with requests.head(url, allow_redirects=False, timeout=(15, 45)) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Official source HEAD failed: HTTP {response.status_code}: {url}")
            size = int(response.headers.get("Content-Length", 0))
            if not 0 < size <= FILE_LIMIT:
                raise ValueError(f"Unexpected bounded source size: {size}: {url}")
            result[kind + "_bytes"] = size
            result[kind + "_last_modified"] = response.headers.get("Last-Modified")
    return result


def probe_plan(plan, workers=4):
    """Add exact transfer estimates; this function never fetches photo bodies."""
    if not 1 <= workers <= 4:
        raise ValueError("Source preflight supports one to four workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(probe_pair_sizes, plan["records"]))
    pair_sizes = [row["input_bytes"] + row["target_bytes"] for row in records]
    return dict(
        plan,
        records=records,
        source_bytes=sum(pair_sizes),
        largest_pair_bytes=max(pair_sizes),
        size_estimate_method="Exact official HEAD Content-Length",
    )


def download_pair(record, stage, max_bytes, *, min_free_bytes=MIN_FREE):
    """Download one pair with strict size limits, SHA256 receipts and resume checks.

    ``stage`` must be a per-scene temporary directory. Return source paths and
    digests; the caller prepares its cache and removes this directory afterward.
    Existing files are reused only with a matching prior SHA256 receipt.
    Published source hashes are unavailable: first-download SHA256 records the
    actual received bytes, and subsequent acquisitions must retain provenance.
    """
    root = Path(stage).resolve()
    if not isinstance(max_bytes, int) or max_bytes <= 0 or min_free_bytes < 0:
        raise ValueError("Require positive byte budget and nonnegative disk reserve")
    for kind in ("input", "target"):
        _validate_source_url(record[kind + "_url"], kind)
        if not 0 < record[kind + "_bytes"] <= FILE_LIMIT:
            raise ValueError("Source size absent or outside individual file budget")
    pending_bytes = sum(
        record[kind + "_bytes"]
        for kind in ("input", "target")
        if not (root / Path(urlparse(record[kind + "_url"]).path).name).exists()
    )
    if pending_bytes > max_bytes:
        raise OSError("Pair download would exceed the configured network budget")
    guard = StorageGuard(root, min_free=min_free_bytes, max_staging=2 * FILE_LIMIT + 1024**2)
    guard.check()
    network_bytes = 0
    result = dict(record)
    for kind in ("input", "target"):
        url, expected = record[kind + "_url"], record[kind + "_bytes"]
        destination = root / Path(urlparse(url).path).name
        receipt_path = destination.with_suffix(destination.suffix + ".source.json")
        if destination.exists() or receipt_path.exists():
            if not destination.is_file() or not receipt_path.is_file():
                raise ValueError(f"Incomplete existing source receipt: {destination}")
            receipt = json.loads(receipt_path.read_text())
            digest = sha256_file(destination)
            if (
                receipt.get("url") != url
                or receipt.get("bytes") != expected
                or destination.stat().st_size != expected
                or receipt.get("sha256") != digest
            ):
                raise ValueError(f"Source receipt/checksum mismatch: {destination}")
        else:
            if network_bytes + expected > max_bytes:
                raise OSError("Pair download would exceed the configured network budget")
            guard.check(expected)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            digest_state = hashlib.sha256()
            received = 0
            try:
                with requests.get(
                    url,
                    stream=True,
                    allow_redirects=False,
                    headers={"Accept-Encoding": "identity"},
                    timeout=(15, 90),
                ) as response:
                    if (
                        response.status_code != 200
                        or int(response.headers.get("Content-Length", 0)) != expected
                    ):
                        raise RuntimeError("Official source status/size differs from preflight")
                    prior_version = record.get(kind + "_last_modified")
                    if prior_version and response.headers.get("Last-Modified") != prior_version:
                        raise RuntimeError("Official source changed after preflight")
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(1024**2):
                            received += len(block)
                            network_bytes += len(block)
                            if received > expected or network_bytes > max_bytes:
                                raise OSError("Source download exceeded its bounded byte budget")
                            stream.write(block)
                            digest_state.update(block)
                    if received != expected:
                        raise OSError("Truncated official source download")
                digest = digest_state.hexdigest()
                expected_hash = record.get(kind + "_sha256")
                if expected_hash and digest != expected_hash:
                    raise ValueError("Source hash differs from pinned source digest")
                with temporary.open("rb") as stream:
                    if stream.read(4) not in (b"II*\x00", b"MM\x00*"):
                        raise ValueError("Expected classic TIFF/DNG source signature")
                temporary.replace(destination)
                _atomic_json(receipt_path, {"url": url, "bytes": expected, "sha256": digest})
            finally:
                temporary.unlink(missing_ok=True)
        if record.get(kind + "_sha256") and record[kind + "_sha256"] != digest:
            raise ValueError("Existing source differs from pinned source digest")
        result[kind + "_path"] = str(destination)
        result[kind + "_sha256"] = digest
    result["network_bytes"] = network_bytes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/splits/cloud_full_7030.json")
    parser.add_argument("--output", default="data/raw_fivek_plan")
    parser.add_argument("--index", help="Previously downloaded official HTML; otherwise fetch bounded index")
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--val-count", type=int, default=32)
    parser.add_argument("--probe-sizes", action="store_true", help="HEAD each pair; still no photo downloads")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    index_html = Path(args.index).read_text() if args.index else fetch_official_index(root)
    plan = plan_fivek_raw(args.registry, index_html, args.train_count, args.val_count)
    if args.probe_sizes:
        plan = probe_plan(plan)
    _retain_fivek_terms(root)
    _atomic_json(root / "plan.json", plan)
    print(json.dumps({key: value for key, value in plan.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
