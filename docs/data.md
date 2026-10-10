# Data artifacts and preparation

The local datasets below are acquired research data, not generated fixtures. All manifest hashes, scene splits, image checksums and paired dimensions have been checked. The original FiveK contains 5,000 photographs; our **2,748-scene collection is the subset recoverable from the mirror's supplied split lists**, not the complete official benchmark.

| Local manifest | Unique scenes | Train / validation / test rows | Styles |
|---|---:|---:|---|
| FiveK collection (generated locally: `../data/fivek/manifest.jsonl`) | 2,748 | 2,025 / 225 / 498 | Expert C |
| Frozen development subset (generated locally: `../data/fivek/manifest.preview.jsonl`) | 192 | 128 / 32 / 32 | Expert C |
| Initial pipeline pilot (generated locally: `../data/fivek_pilot/manifest.jsonl`) | 24 | 16 / 4 / 4 | Expert C |
| Three-editor pilot (generated locally: `../data/fivek_multistyle/manifest.jsonl`) | 24 | 48 / 12 / 12 | 0 = C, 1 = A, 2 = B |

The three-editor pilot contains the first 16 training, four validation and four test scenes, sorted by scene ID, from the frozen development subset. Each scene has three targets; its split stays unchanged. The 192-scene manifest was frozen from the first complete locally downloaded pairs. Its exact selection is now versioned in [the preview registry](../configs/splits/fivek_preview.json); [the three-editor registry](../configs/splits/fivek_multistyle.json) freezes the 24-scene C/A/B selection. Both registries retain ordered scene/style rows, original manifest SHA256, source provenance and image digests, with no images. These are development subsets, not published evaluation protocols. The initial 24-scene pilot is a separate selection under the same split mapping. Manifest paths are relative to the manifest; the three-editor manifest references existing FiveK inputs and C targets, so both directories are needed. Data files and generated reports are excluded from Git. The tracked registries and `photograder.splits` helper reconstruct/check the ignored manifests after acquisition. The helper rejects missing scenes, changed splits/styles/rendering metadata, changed file bytes and a different canonical manifest hash; it never overwrites an existing output. Keep rebuilt manifests beside their source manifest so relative paths remain unchanged. The prepared FiveK JPEG subset can reconstruct byte-identically from the same archive members. For native A/B conversion, the registries preserve original hashes as evidence and fix the scene/style/split protocol; they do not guarantee byte-identical assets on another machine. ColorSync, Pillow, ICC profile timestamps and encoder metadata can change a file hash even when decoded pixels agree. Such mismatches are rejected clearly; record regenerated assets as a separate dataset rendering rather than replacing the frozen evidence.

## Sources, integrity and storage

FiveK inputs and Expert C targets come from the [Image-Adaptive-3DLUT release](https://github.com/HuiZeng/Image-Adaptive-3DLUT), using the author's documented `input/JPG/480p` and `expertC/JPG/480p` layout. Collection provenance (generated locally: `../data/fivek/provenance.json`) and initial-pilot provenance (generated locally: `../data/fivek_pilot/provenance.json`) record the mirror, archive version and split-file hashes.

The source ZIP is **12,252,167,420 bytes**. HTTP ranges retrieved its central directory and only selected JPEG members; the full archive was not downloaded. Selected collection JPEGs occupy **1,914,747,961 bytes** before filesystem overhead. The mirror supplies 2,250 training IDs and 498 test IDs but omits `train_label.txt`; missing training IDs were not guessed. Validation takes the lowest 225 values of `SHA256("fivek-val-v1:" + photo_id)` from the supplied training list. Published test IDs remain held out.

The downloader requires exact HTTP 206 ranges and archive size, bounds reads, checks ZIP CRC32 and payload sizes, refuses unsafe member names, and verifies existing files before resuming. Manifests record per-file SHA256. No complete-archive SHA256 is claimed. Storage guards preserve **20 GiB free** and cap each destination's staging at **20 GiB**. The initial pilot has a 100 MiB network budget. Network counters describe their recorded invocation; the collection's final counter excludes its earlier interrupted pass.

Original [FiveK research licenses](https://data.csail.mit.edu/graphics/fivek/) and their file-applicability lists are retained beside the data as `LicenseAdobe.txt`, `LicenseAdobeMIT.txt`, `filesAdobe.txt`, and `filesAdobeMIT.txt`. They restrict use to research without commercial advantage. Source attribution and license terms remain applicable to derived artifacts.

## Additional editors and alignment audit

The three-editor provenance (generated locally: `../data/fivek_multistyle/provenance.json`) records **48 individually downloaded official A/B TIFFs**, their URLs and SHA256 values, and **2,214,803,612 bytes** of transfer including the official index, below the 3 GiB budget. One source TIFF and its temporary converted TIFF were staged at a time and deleted after the final PNG and provenance were saved.

The official targets are 16-bit ProPhoto RGB. macOS `sips`/ColorSync transforms the original 16-bit pixels into 8-bit sRGB before Pillow decoding, then LANCZOS resizes them to the existing paired dimensions. Output render version is `fivek-colorsync16input-srgb8-lanczos-v1`. Source aspect ratios must agree within one output pixel of rounding; a larger discrepancy aborts import. A precision probe varies only the low eight bits of 16-bit source samples: native ColorSync output varied in **47 of 64 rows**, versus zero rows after prematurely quantizing the source in Pillow. The importer now runs this check dynamically; it does not silently fall back on platforms without ColorSync.

This produces 8-bit sRGB PNG targets, not lossless 16-bit working masters. Expert C remains the prepared JPEG release, so resampling/compression and rendering paths differ between C and A/B. Editor IDs represent real retouching preferences rather than named film aesthetics. The 24-scene style corpus is suitable for an implementation pilot, not a generalization claim.

Three input/C/A/B contact sheets were visually inspected: building, a0040 (generated locally: `../data/fivek_multistyle/alignment_audit/1_fivek_a0040.png`), rose, a0041 (generated locally: `../data/fivek_multistyle/alignment_audit/2_fivek_a0041.png`), and animal/fence, a0069 (generated locally: `../data/fivek_multistyle/alignment_audit/3_fivek_a0069.png`). Building edges, petals, ears and fence intersections occupy matching positions; no obvious crop, rotation or translation discrepancy was found. These are two training scenes and one validation scene, inspected for geometry only.

An integer translation search over ±3 pixels compared luminance-gradient magnitude against each input, excluding an eight-pixel border. All nine comparisons peaked at **(0, 0)**; normalized edge correlations ranged from **0.859 to 0.991**. Audit details (generated locally: `../data/fivek_multistyle/alignment_audit/report.json`) retain the per-scene values. This sample check does not certify subpixel alignment, all 24 scenes, or arbitrary future imports.

## S24, loader behavior and commands

**No usable S24 dataset was acquired, and no S24 data was used in training.** A browser check on 2026-10-01 verified the official training share's inventory: **15 split-archive parts**, `train.z01` through `train.z14` plus `train.zip`. The first part's preview displays **20.0 GB** (rounded service display; exact bytes and total archive size remain unverified). There are no individual photographs in that visible listing. **The acquisition gate remains external storage or verified selective extraction from this multipart share.** Neither bounded per-photo access nor validation/test archive sizes were established.

The share’s “Open in browser” action unexpectedly initiated a partial archive transfer. At **06:42 UTC on 2026-10-01**, a narrow check for its known inode and download filenames found no matches; the previously observed partial file was absent. Free space was **23.341 GiB**, restoring the unchanged **20 GiB reserve**. The cause of the file's disappearance was not established; this report does not assert an explicit user cancellation reply or programmatic cancellation. No partial archive was imported or used for training, and its internal schema remains unverified. One complete 20.0 GB part would still exceed the available space above the reserve. Earlier bounded HTTP requests returned only the HTML landing page; the later browser inventory and unexpected partial transfer are separate events. Access-check record (generated locally: `../data/s24_access_probe.json`)

Its [official structure](https://github.com/SamsungLabs/time-aware-awb/blob/main/s24-raw-srgb-dataset/README.md) supports a local importer using genuine 16-bit denoised RAW PNGs, camera-estimated white balance, CCM and the standard sRGB transfer. Ground-truth white balance is not used. Camera JPEGs are excluded because the authors document a pixel shift. `blur_masks > 0` marks invalid privacy regions; secondary-illuminant masks are not correction labels. Preserve the [source license](https://github.com/SamsungLabs/time-aware-awb/blob/main/LICENSE.md), which has non-commercial terms.

The importer retains official train/validation/test folders and records orientation policy and pre-clamp gamut statistics. Selected source styles map to contiguous model IDs in the requested order: `--styles 2,5` produces model IDs 0 and 1, names “S24 Style 2” and “S24 Style 5”, and retained `source_style_id` values 2 and 5. Actual S24 geometry, metadata conventions and rendering quality still require visual verification once files are supplied. The implementation has passed 16-bit fixture, privacy-mask and noncontiguous-style tests; that is not validation on the real S24 dataset.

The shared loader honors embedded ICC profiles and EXIF orientation, assumes sRGB only when no profile is supplied, rejects mismatched paired dimensions, and preserves aspect ratio using edge padding. Padding, alpha transparency and optional privacy masks are excluded from losses and metrics. Invalid embedded profiles raise an error. Every scene's edits must share a split; identical image content across splits is also rejected.

Run from the project root:

```bash
# Existing artifacts: checksum, scene-split and paired-geometry verification.
.venv/bin/python -m photograder verify-data data/fivek/manifest.jsonl

# Reacquire the bounded initial pilot or all pairs supplied by the mirror lists.
.venv/bin/python -m photograder.acquisition fivek --output data/fivek_pilot
.venv/bin/python -m photograder.acquisition fivek --output data/fivek --full --workers 24

# Fresh clone: reconstruct the exact192-scene manifest from acquired full pairs.
# For an existing manifest, replace --output PATH with --verify-only.
.venv/bin/python -m photograder.splits --registry configs/splits/fivek_preview.json --source data/fivek/manifest.jsonl --output data/fivek/manifest.preview.jsonl

# macOS only: acquire the exact16/4/4-scene A/B extension, then check its registry.
.venv/bin/python -m photograder.acquisition fivek-experts --manifest data/fivek/manifest.preview.jsonl --output data/fivek_multistyle
.venv/bin/python -m photograder.splits --registry configs/splits/fivek_multistyle.json --source data/fivek_multistyle/manifest.jsonl --verify-only

# Optional high-precision PNG decoder, then import locally supplied S24 files.
uv pip install --python .venv/bin/python -e '.[data]'
.venv/bin/python -m photograder.acquisition s24 --source /path/to/S24 --output data/s24 --styles 0,1,2 --raw-orientation metadata
```

Use a fresh destination for S24 rendering; existing outputs are not silently overwritten. `--raw-orientation already_oriented` is only appropriate when source RAW and masks have already been normalized. Large-file acquisition commands transfer real data and retain the documented budgets; validation and existing-manifest use need no new downloads.
