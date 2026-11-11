# RAW data preparation

[FiveK](https://data.csail.mit.edu/graphics/fivek/) provides individually downloadable original DNGs and 16-bit ProPhoto RGB expert TIFFs. The acquisition module pairs those URLs, checks file sizes/signatures, and records SHA256 receipts. CPU preparation streams one scene at a time into a float cache.

## Pilot selection

The [original 70/30 registry](../configs/splits/cloud_full_7030.json) supplies 1,924 outer-training scenes. SHA256 ranking reserves 192 for inner validation; 1,732 remain for inner training. A second fixed ranking chooses nested acquisition subsets, preserving membership as the set grows.

The initial 128/32 request transfers **10,210,570,990 bytes**, about 9.51 GiB. The largest pair is about 135.9 MiB. After preparation checks, 123 training and 32 validation scenes remain. [Pilot results](reports/raw.md) record the exclusions and training outcomes.

## Prepare and train

Install RAW/cloud dependencies and LittleCMS2 as described in [usage](usage.md#raw-developer). Create the source plan:

```bash
uv run --no-sync python -m photograder.raw_acquisition \
  --registry configs/splits/cloud_full_7030.json \
  --output data/raw_fivek_plan \
  --train-count 128 --val-count 32 --probe-sizes
```

This step reads source metadata and sizes. CPU preparation then downloads each pair, creates the cache, and removes its temporary originals:

```bash
uv run --no-sync modal run --detach modal_raw_train.py --prepare-only
```

After preparation completes, launch the four-arm pilot with a fresh run ID:

```bash
uv run --no-sync modal run --detach modal_raw_train.py --run-id raw-run1
```

The data volume retains source receipts, preparation metadata, masks, and float arrays. Downloaded originals are temporary staging files. Each cache records its decoder/rendering version and exact geometry.

## Geometry and color

Training inputs are float32 scene-linear sRGB. Targets are float32 display sRGB converted through LittleCMS from the embedded ProPhoto profile. DNG `ActiveArea`, `DefaultCropOrigin`, and `DefaultCropSize` define the crop before orientation.

| Scene | Sensor W × H | ActiveArea `[top,left,bottom,right]` | Crop origin | Crop size | Orientation |
|---|---|---|---|---|---:|
| a0001 | 3040 × 2014 | `[0,0,2014,3040]` | `[15,7]` | `[3008,2000]` | 1 |
| a0005 | 3596 × 2360 | `[12,74,2360,3596]` | `[10,7]` | `[3504,2336]` | 8 |
| a0007 | 4312 × 2876 | `[18,22,2874,4312]` | `[10,5]` | `[4272,2848]` | 6 |

These crops match the oriented expert targets. The metadata reader follows sensor SubIFDs, handles rational tags, and rejects fractional crops or inconsistent rectangles. Crop placement, orientation, and paired dimensions are checked before caching. See [raw_acquisition.py](../src/photograder/raw_acquisition.py) and the [decoder audit](raw-pipeline-research.md).

## S24

The [S24 dataset](https://github.com/SamsungLabs/time-aware-awb/blob/main/s24-raw-srgb-dataset/README.md) includes 3,224 scenes, denoised RAW RGB PNGs, Lightroom styles, and capture metadata. Acquisition is pending: the inspected training share used 15 multipart archive files, with the first displayed as 20.0 GB.

For a future RAW experiment, preserve the linear color conversion before display clipping, use camera-estimated white balance, retain official splits, and exclude privacy blur masks. Camera JPEGs have a documented spatial shift. Secondary-illuminant masks describe lighting regions.

FiveK's licenses and per-file applicability lists are retained with the source plan. S24 carries its own [license](https://github.com/SamsungLabs/time-aware-awb/blob/main/LICENSE.md). The repository distributes code, reports, and split registries; source datasets are acquired separately.
