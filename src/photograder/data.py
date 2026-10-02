"""Color-managed paired data and scene-safe JSONL manifests.

Manifests use paths relative to their own directory. ``valid_mask`` is optional
and uses 1/255 for valid pixels (privacy masks must be inverted on import).
All versions of a scene, including different styles, must share one split.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps
from torch.utils.data import Dataset

RENDER_VERSION = "srgb-exif-v1"
REQUIRED = {
    "source_id",
    "scene_id",
    "input",
    "target",
    "style_id",
    "style_name",
    "split",
    "sha256",
    "render_version",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_srgb(path):
    """Return an EXIF-oriented RGB PIL image, honoring its ICC profile.

    Invalid embedded profiles raise rather than silently assuming sRGB. Inputs
    without an ICC profile are assumed sRGB; that assumption is versioned in the
    manifest. Alpha is handled separately as training validity, not composited.
    """
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        profile = source.info.get("icc_profile")
        image = image.convert("RGB")
        if profile:
            image = ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        return image.copy()


def _alpha(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if "A" in image.getbands() or "transparency" in source.info:
            return np.asarray(image.convert("RGBA"), dtype=np.float32)[..., 3] / 255.0
        return np.ones((image.height, image.width), dtype=np.float32)


def read_manifest(path):
    path = Path(path)
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{number}") from error
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"Manifest row {number} missing {sorted(missing)}")
            if row["split"] not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split in row {number}")
            if not isinstance(row["style_id"], int) or row["style_id"] < 0:
                raise ValueError(f"Invalid style_id in row {number}")
            if not all(
                isinstance(row[k], str) and row[k]
                for k in ("source_id", "scene_id", "input", "target", "style_name", "render_version")
            ):
                raise ValueError(f"Empty or invalid manifest fields at row {number}")
            if not isinstance(row["sha256"], dict) or not {"input", "target"}.issubset(row["sha256"]):
                raise ValueError("sha256 must contain input and target digests")
            rows.append(row)
    _validate_splits(rows)
    return rows


def _validate_splits(rows):
    scene_splits, hash_splits, styles, pairs = {}, {}, {}, set()
    for row in rows:
        scene, split = row["scene_id"], row["split"]
        if scene in scene_splits and scene_splits[scene] != split:
            raise ValueError(f"Scene leakage across splits: {scene}")
        scene_splits[scene] = split
        key = (scene, row["style_id"], row["render_version"], row["input"])
        if key in pairs:
            raise ValueError(f"Duplicate scene/style/render/input: {key}")
        pairs.add(key)
        if row["style_id"] in styles and styles[row["style_id"]] != row["style_name"]:
            raise ValueError("One style_id maps to multiple style names")
        styles[row["style_id"]] = row["style_name"]
        for kind, digest in row["sha256"].items():
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid SHA256 digest")
            if kind in {"input", "target"}:
                if digest in hash_splits and hash_splits[digest] != split:
                    raise ValueError("Identical image content occurs in different splits")
                hash_splits[digest] = split


def validate_manifest(path, verify_hashes=True, verify_geometry=True):
    """Validate all rows (including splits not requested for training)."""
    path = Path(path)
    rows = read_manifest(path)
    checked = {}
    for row in rows:
        for key in ("input", "target", "valid_mask"):
            if key not in row:
                continue
            image_path = (path.parent / row[key]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if verify_hashes:
                if image_path not in checked:
                    checked[image_path] = sha256_file(image_path)
                expected = row["sha256"].get(key)
                # Reuse the computed digest, never a previous row's validation.
                # Every row must agree, including aliases and repeated style inputs.
                if not expected or checked[image_path] != expected:
                    raise ValueError(f"Checksum mismatch or missing checksum: {image_path}")
        if verify_geometry:
            left, right = load_srgb(path.parent / row["input"]), load_srgb(path.parent / row["target"])
            if left.size != right.size:
                raise ValueError(
                    f"Paired dimensions mismatch: {row['scene_id']}: {left.size} != {right.size}"
                )
            if "valid_mask" in row:
                with Image.open(path.parent / row["valid_mask"]) as mask:
                    if ImageOps.exif_transpose(mask).size != left.size:
                        raise ValueError(f"Mask dimensions mismatch: {row['scene_id']}")
    return {
        "rows": len(rows),
        "scenes": len({r["scene_id"] for r in rows}),
        "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "styles": {r["style_id"]: r["style_name"] for r in rows},
    }


def write_manifest(rows, path):
    """Write atomically; never reassign an existing scene to a different split."""
    path = Path(path)
    _validate_splits(rows)
    if path.exists():
        old = {r["scene_id"]: r["split"] for r in read_manifest(path)}
        for row in rows:
            if row["scene_id"] in old and old[row["scene_id"]] != row["split"]:
                raise ValueError(f"Existing scene split is immutable: {row['scene_id']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


class PairedDataset(Dataset):
    """Load square padded pairs, preserving aspect ratio and masking padding.

    Returns image/target float32 [3,S,S], valid_mask [1,S,S], style_id int64,
    scene_id str, original_size int64 [width,height], and content_box int64
    [left,top,width,height]. ``limit`` selects rows after whole-manifest leakage
    validation. For full integrity checking call validate_manifest once first.
    """

    def __init__(self, manifest, split, image_size=512, limit: int | None = None):
        self.manifest = Path(manifest)
        if image_size < 16 or split not in {"train", "val", "test"}:
            raise ValueError("image_size must be >=16 and split train/val/test")
        self.image_size = int(image_size)
        self.rows = [r for r in read_manifest(manifest) if r["split"] == split]
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No {split} examples in {manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        base = self.manifest.parent
        left, right = load_srgb(base / row["input"]), load_srgb(base / row["target"])
        if left.size != right.size:
            raise ValueError(f"Unaligned pair dimensions: {row['scene_id']}")
        width, height = left.size
        valid = _alpha(base / row["input"]) * _alpha(base / row["target"])
        if "valid_mask" in row:
            with Image.open(base / row["valid_mask"]) as source:
                mask = np.asarray(ImageOps.exif_transpose(source).convert("L"), dtype=np.float32)
            if mask.shape != valid.shape:
                raise ValueError("Validity mask dimensions differ from image")
            valid *= (mask > 0).astype(np.float32)
        size = self.image_size
        scale = min(size / width, size / height, 1.0)
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        x, y = (size - new_w) // 2, (size - new_h) // 2
        pad = ((y, size - new_h - y), (x, size - new_w - x), (0, 0))
        tensors = []
        for image in (left, right):
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = np.pad(array, pad, mode="edge")
            tensors.append(torch.from_numpy(array.transpose(2, 0, 1).copy()))
        valid = np.asarray(Image.fromarray(valid).resize((new_w, new_h), Image.Resampling.NEAREST))
        valid = np.pad(valid, (pad[0], pad[1]), mode="constant")
        return {
            "image": tensors[0],
            "target": tensors[1],
            "valid_mask": torch.from_numpy(valid[None].copy()),
            "style_id": torch.tensor(row["style_id"], dtype=torch.long),
            "scene_id": row["scene_id"],
            "original_size": torch.tensor([width, height]),
            "content_box": torch.tensor([x, y, new_w, new_h]),
        }


def render_s24_linear(raw_rgb, metadata):
    """Fixed display rendering: camera illuminant normalization, CCM, sRGB.

    raw_rgb must be *genuine* normalized floating RGB from the 16-bit denoised
    PNG, without automatic gamma or quantization. Ground-truth WB is never used.
    Returns an sRGB float32 image and pre-clamp out-of-gamut fraction.
    """
    raw = np.asarray(raw_rgb)
    if not np.issubdtype(raw.dtype, np.floating) or raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("Expected normalized float HWC RAW RGB; decode PNG at 16-bit precision")
    if not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise ValueError("RAW must be finite and normalized to [0,1]")
    illuminant = np.asarray(metadata["cam_illum"], dtype=np.float32).reshape(3)
    ccm = np.asarray(metadata["ccm"], dtype=np.float32).reshape(3, 3)
    if not np.isfinite(illuminant).all() or (illuminant <= 0).any() or not np.isfinite(ccm).all():
        raise ValueError("Invalid camera illuminant or CCM")
    gain = illuminant[1] / illuminant
    linear = np.einsum("...c,dc->...d", raw.astype(np.float32) * gain, ccm)
    gamut = float(((linear < 0) | (linear > 1)).any(axis=-1).mean())
    linear = np.clip(linear, 0, 1)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return srgb.astype(np.float32), gamut


def _orient_array(array, orientation):
    """Apply EXIF 1..8 to HWC or HW data without quantizing RAW values."""
    operations = {
        1: lambda x: x,
        2: lambda x: x[:, ::-1],
        3: lambda x: x[::-1, ::-1],
        4: lambda x: x[::-1],
        5: lambda x: np.swapaxes(x, 0, 1),
        6: lambda x: np.rot90(x, -1),
        7: lambda x: np.swapaxes(x[::-1, ::-1], 0, 1),
        8: lambda x: np.rot90(x, 1),
    }
    if orientation not in operations:
        raise ValueError(f"Unsupported orientation {orientation}; expected EXIF integer1..8")
    return operations[orientation](array).copy()


def _read_rgb16(path):
    try:
        import png
    except ImportError as error:
        raise RuntimeError(
            "S24 preparation requires pypng: pip install 'selective-photograder[data]'"
        ) from error
    width, height, rows, info = png.Reader(filename=str(path)).asDirect()
    if info["bitdepth"] != 16 or info["planes"] != 3 or info.get("greyscale") or info.get("alpha"):
        raise ValueError(f"Expected genuine 16-bit RGB PNG: {path}")
    array = np.vstack([np.asarray(row, dtype=np.uint16) for row in rows]).reshape(height, width, 3)
    return array.astype(np.float32) / 65535.0


def import_s24(source, output, styles=(0, 1, 2), limit_per_split=None, raw_orientation="metadata"):
    """Prepare a *locally supplied* official S24 directory; no access bypass.

    Expected: train/val/test/{denoised_raw_images,data,srgb_images_style_N,
    blur_masks}/ID.{png,json,jpg,png}. Camera JPEGs are deliberately not used.
    ``raw_orientation='metadata'`` applies top-level integer EXIF orientation,
    default1 when absent, to RAW and its blur mask; ``already_oriented`` is for
    sources whose pixel orientation has already been normalized. The chosen
    policy is recorded in each manifest. Selected source styles map to contiguous
    model IDs in the requested order; original numbers remain source_style_id.
    Always visually audit alignment.
    """
    from .acquisition import StorageGuard

    source, output = Path(source).resolve(), Path(output).resolve()
    if raw_orientation not in {"metadata", "already_oriented"}:
        raise ValueError("raw_orientation must be metadata or already_oriented")
    styles = tuple(int(s) for s in styles)
    style_to_model = {source_style: model_style for model_style, source_style in enumerate(styles)}
    if not styles or any(s < 0 or s > 5 for s in styles) or len(set(styles)) != len(styles):
        raise ValueError("Styles must be unique S24 IDs0..5")
    guard = StorageGuard(output)
    guard.check()
    manifest_rows = []
    for split in ("train", "val", "test"):
        raw_dir = source / split / "denoised_raw_images"
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Missing S24 source split: {raw_dir}")
        paths = sorted(raw_dir.glob("*.png"))
        if limit_per_split is not None:
            paths = paths[: int(limit_per_split)]
        for raw_path in paths:
            identity = raw_path.stem
            metadata_path = source / split / "data" / (identity + ".json")
            metadata = json.loads(metadata_path.read_text())
            raw = _read_rgb16(raw_path)
            orientation = metadata.get("orientation", 1) if raw_orientation == "metadata" else 1
            if not isinstance(orientation, int):
                raise ValueError(f"Non-integer S24 orientation: {metadata_path}")
            raw = _orient_array(raw, orientation)
            rendered, gamut = render_s24_linear(raw, metadata)
            blur_path = source / split / "blur_masks" / (identity + ".png")
            if not blur_path.is_file():
                raise FileNotFoundError(f"Privacy blur mask required: {blur_path}")
            with Image.open(blur_path) as image:
                invalid = _orient_array(np.asarray(image.convert("L")) > 0, orientation)
            if invalid.shape != rendered.shape[:2]:
                raise ValueError(f"S24 blur mask dimensions do not match RAW: {identity}")
            # Conservative budget for all uncompressed outputs before each photo.
            guard.check(rendered.shape[0] * rendered.shape[1] * (4 + 3 * len(styles)))
            inputs = output / "input"
            masks = output / "valid_masks"
            inputs.mkdir(parents=True, exist_ok=True)
            masks.mkdir(parents=True, exist_ok=True)
            input_path, mask_path = inputs / (identity + ".png"), masks / (identity + ".png")
            if input_path.exists():
                raise FileExistsError(
                    f"Output exists; use a fresh destination to preserve rendering provenance: {input_path}"
                )
            Image.fromarray(np.round(rendered * 255).astype(np.uint8)).save(input_path)
            Image.fromarray((~invalid).astype(np.uint8) * 255).save(mask_path)
            for style in styles:
                target_path = source / split / f"srgb_images_style_{style}" / (identity + ".jpg")
                target = load_srgb(target_path)
                if target.size != (rendered.shape[1], rendered.shape[0]):
                    raise ValueError(
                        f"S24 target geometry differs after normalization: {identity}, style{style}"
                    )
                target_dir = output / f"style_{style}"
                target_dir.mkdir(parents=True, exist_ok=True)
                saved_target = target_dir / (identity + ".png")
                target.save(saved_target)
                row = {
                    "source_id": "S24-Raw-sRGB",
                    "scene_id": "s24:" + identity,
                    "split": split,
                    "style_id": style_to_model[style],
                    "style_name": f"S24 Style {style}",
                    "source_style_id": style,
                    "input": str(input_path.relative_to(output)),
                    "target": str(saved_target.relative_to(output)),
                    "valid_mask": str(mask_path.relative_to(output)),
                    "render_version": f"s24-camwb-ccm-srgb-v1-{raw_orientation}",
                    "source_raw_sha256": sha256_file(raw_path),
                    "source_metadata_sha256": sha256_file(metadata_path),
                    "preclamp_out_of_gamut_fraction": gamut,
                    "raw_orientation_applied": orientation,
                }
                row["sha256"] = {
                    key: sha256_file(output / row[key]) for key in ("input", "target", "valid_mask")
                }
                manifest_rows.append(row)
    manifest = output / "manifest.jsonl"
    write_manifest(manifest_rows, manifest)
    provenance = {
        "dataset": "S24 Raw-sRGB",
        "acquisition": "Imported locally supplied official source files; no automatic remote access",
        "documentation": "https://github.com/SamsungLabs/time-aware-awb/blob/main/s24-raw-srgb-dataset/README.md",
        "terms_url": "https://github.com/SamsungLabs/time-aware-awb/blob/main/LICENSE.md",
        "usage": "Research/non-commercial terms apply; retain original LICENSE. No source dataset redistribution.",
        "rendering": "Denoised16bit RGB /65535; gains=cam_illum.green/cam_illum; output=CCM@WB_RGB; clamp linear; standard sRGB transfer;8bit PNG. No target WB used.",
        "orientation_policy": raw_orientation,
        "model_to_source_style": {model_id: source_id for source_id, model_id in style_to_model.items()},
        "alignment": "Dimensions checked against expert targets; visual alignment audit still required. Camera JPEGs excluded due documented pixel shift.",
        "privacy": "blur_masks>0 excluded; secondary-illuminant masks are not used as correction labels.",
        "split_policy": "Official train/val/test folders retained; scene leakage validation applied globally.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    validate_manifest(manifest)
    return manifest
