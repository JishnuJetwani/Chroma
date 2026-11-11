# Data preparation

The JPEG experiments use [MIT–Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/), with Expert C as the main target. The latest run covers **all 5,000 scenes**. Inputs and C targets come from the prepared 480p [Image-Adaptive 3D LUT release](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

| Experiment | Scenes | Train / validation / test scenes | Registry |
|---|---:|---:|---|
| Full FiveK | 5,000 | 3,500 / 0 / 1,500 | [Full split](../configs/splits/fivek_5000_7030.json) |
| Earlier 70/30 run | 2,748 | 1,924 / 0 / 824 | [Subset split](../configs/splits/cloud_full_7030.json) |
| Development pilot | 192 | 128 / 32 / 32 | [Preview split](../configs/splits/fivek_preview.json) |
| Three-editor pilot | 24 | 16 / 4 / 4 | [Editor split](../configs/splits/fivek_multistyle.json) |

The three-editor pilot has C/A/B targets for each scene. Alternate edits, crops, and synthetic variants stay with their source photo. Both 70/30 runs use custom splits.

## Acquire JPEG pairs

Run from the project root after the [environment setup](usage.md#setup):

```bash
# 16 training, four validation, and four test scenes.
uv run --no-sync photograder acquire --output data/fivek_pilot

# All 2,748 scenes listed in the prepared release's split files.
uv run --no-sync photograder acquire --full --output data/fivek --workers 8
uv run --no-sync photograder verify-data data/fivek/manifest.jsonl
```

The `--full` flag covers the supplied lists: 2,250 training IDs and 498 test IDs. The importer reserves 225 training IDs for validation. Acquiring the remaining 2,252 scenes uses the separate [full FiveK workflow](full-fivek-training.md).

The downloader reads the ZIP index and selected JPEG members through bounded HTTP ranges. It checks CRC32, sizes, SHA256, and paired geometry, and keeps a source manifest. Local storage guards retain 20 GiB free and cap staging at 20 GiB. The 2,748-scene JPEG collection occupies about 1.91 GB; the complete archive is 12.25 GB.

## Rebuild the development split

After acquiring `data/fivek`, reconstruct the tracked 192-scene selection:

```bash
uv run --no-sync python -m photograder.splits \
  --registry configs/splits/fivek_preview.json \
  --source data/fivek/manifest.jsonl \
  --output data/fivek/manifest.preview.jsonl
```

The helper checks scene IDs, splits, styles, rendering metadata, and source hashes. Use `--verify-only` to check an existing manifest. Keep rebuilt manifests beside their source manifest so relative image paths resolve.

## Additional editors

The macOS importer downloads official 16-bit ProPhoto RGB A/B TIFFs, converts them through ColorSync, and resizes them to the paired JPEG geometry:

```bash
uv run --no-sync python -m photograder.acquisition fivek-experts \
  --manifest data/fivek/manifest.preview.jsonl --output data/fivek_multistyle
uv run --no-sync python -m photograder.splits \
  --registry configs/splits/fivek_multistyle.json \
  --source data/fivek_multistyle/manifest.jsonl --verify-only
```

This produces 24 scenes with three editor targets each. The 70/30 runs assign 17 of those scenes to training and seven to testing. A/B outputs are 8-bit sRGB PNGs. Native color-conversion and encoder versions can change file hashes; regenerated renderings should have their own provenance.

Three inspected scenes showed aligned edges and zero best integer translation in a ±3-pixel search. The shared loader checks paired dimensions, honors EXIF/ICC metadata, and excludes padding, transparency, and privacy masks from losses and metrics.

## S24

S24 acquisition is pending. Its importer accepts locally supplied denoised 16-bit RAW RGB PNGs, applies camera white balance and CCM, and prepares sRGB training inputs. It preserves the official splits and excludes privacy-blurred pixels. The camera JPEGs have a documented spatial shift, so the importer uses RAW-derived inputs.

```bash
uv run --no-sync python -m photograder.acquisition s24 \
  --source /path/to/S24 --output data/s24 --styles 0,1,2 \
  --raw-orientation metadata
```

Replace `/path/to/S24` with the extracted dataset directory. This adapter produces 8-bit display inputs; a future scene-linear S24 experiment needs a float cache before display conversion. See the [dataset structure](https://github.com/SamsungLabs/time-aware-awb/blob/main/s24-raw-srgb-dataset/README.md) and [RAW acquisition notes](raw-data-research.md).

FiveK's research licenses and source attribution are saved during acquisition. S24 retains its [source terms](https://github.com/SamsungLabs/time-aware-awb/blob/main/LICENSE.md). Dataset images and generated caches are excluded from Git.
