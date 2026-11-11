# Chroma

Automatic photo grading with a small CNN, learned LUTs, and local exposure and contrast adjustments. Analyze a photo once, adjust the strength, and export at its original resolution.

**Before → after** · Original on the left, graded result on the right.

**Flower**

![Flower before and after automatic JPEG grading](docs/examples/flower-before-after.png)

**Paris at night**

![Paris at night before and after automatic JPEG grading](docs/examples/paris-before-after.png)

**Rainy street**

![Rainy street before and after, shown in the supplied comparison](docs/examples/rainy-street-before-after.png)

Flower and Paris images are from MIT–Adobe FiveK and show the original JPEG checkpoint. [Image sources and research licenses](docs/examples/README.md). The rainy street is my photo (DC at UWaterloo).

The JPEG model has **1.71 million parameters**. The latest run uses all 5,000 FiveK scenes, split into **3,500 training / 1,500 test photos**, and reaches **24.45 dB PSNR / 0.9038 SSIM**. The previous model scores **24.13 dB / 0.8987** on those same 1,500 photos. [Full-FiveK results and limitations](docs/reports/full-fivek.md).

## Try it

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). CPU, Apple MPS, and NVIDIA CUDA are supported; no custom GPU extensions are needed. 

Model weights aren’t hosted yet. Feel free to email me at jjetwani@uwaterloo.ca for them, or train your own with the steps in Data and training. The full FiveK run takes about 5 minutes on a single H100.

```bash
uv sync --locked --extra app
uv run --no-sync photograder doctor
```

Place the trained JPEG checkpoint at `checkpoints/jpeg.pt`, then run:

```bash
uv run --no-sync photograder app --checkpoint checkpoints/jpeg.pt
```

Open **http://127.0.0.1:7860**, upload a JPEG or PNG, and press **Auto Grade**. The overall-grade and local-correction sliders reuse the cached prediction. Export creates a new file and preserves the original.

See the [model card](docs/model-card.md) for its checksum, training data, and limitations. On macOS, `run_app.command` discovers either location.

To grade a single image without opening the app:

```bash
uv run --no-sync photograder grade photo.jpg \
  --checkpoint checkpoints/jpeg.pt --output exports/photo-graded.png
```

## How it works

1. A four-stage CNN analyzes an aspect-preserving 256-pixel preview.
2. The global head mixes eight learned 33³ RGB lookup tables.
3. A shared spatial decoder predicts exposure and contrast maps with a 64-pixel longest side.
4. The renderer applies the color table and edge-aware local adjustments to the original image. Large exports use overlapping tiles with one shared recipe.

The model predicts **editing operations** rather than replacement image pixels. Exposure is bounded to ±1 stop, and contrast multiplies local log-luminance detail by 0.5–2.0.

The code also includes a fixed LUT baseline, cascaded specialists, an RGB-residual U-Net, and a bilateral-grid alternative as experimental comparisons.

## Data and training

The latest JPEG run uses **all 5,000 FiveK scenes**: 3,500 for training and 1,500 for testing. It expands the original 2,748-scene run while preserving every existing train/test assignment. The main target is Expert C. Editors A/B each still have only 17 training scenes, so those extra styles are exploratory. S24 has an importer but was not used in the reported runs. [Full-data acquisition and training](docs/full-fivek-training.md).

## Results

The original checkpoint, shown in the flower and Paris examples, scored as follows on its 824-photo Expert C holdout:

| Output on Expert C holdout | PSNR ↑ | SSIM ↑ | Color error, CIE76 ↓ |
|---|---:|---:|---:|
| Original input | 18.34 dB | 0.7906 | 16.30 |
| Global LUT model | 21.97 dB | 0.8676 | 10.25 |
| Global + local model | **23.98 dB** | **0.8958** | **8.35** |

Training on the full dataset improves the original 824-photo score to **24.32 dB**, and scores **24.45 dB** across the expanded 1,500-photo holdout. The improvement over the previous model on the same 1,500 photos is **+0.319 dB**, with a paired scene-bootstrap 95% interval of **[+0.233, +0.400]**. Highlight clipping decreases, while shadow clipping increases; more data also means more optimizer updates at the same epoch count. [Full comparison](docs/reports/full-fivek.md).

## Experimental RAW support

A separate RAW pipeline preserves scene-linear highlight information through decoding, grading, and tone mapping. It uses its own model and checkpoint; it is not a drop-in upgrade to the JPEG model.

```bash
uv sync --locked --extra app --extra raw
uv run --no-sync photograder app --raw-checkpoint path/to/raw-model.pt
```

The pilot used 123 training and 32 validation scenes, all drawn from the JPEG training pool. It does not establish a quality improvement over the JPEG model. [RAW report](docs/reports/raw.md) · [Decoder and acquisition notes](docs/raw-direction.md).

## Development

```bash
uv sync --locked --extra app --extra data --extra raw --extra dev --extra cloud
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

Tests cover LUT gradients, identity behavior, tiled rendering, scene-split integrity, checkpoint compatibility, training, and source-preserving exports. Hardware-specific checks skip when their device is unavailable.

- `src/photograder/`: models, renderer, training, inference, and app.
- `configs/`: experiment settings and frozen split registries.
- `tests/`: numerical and integration checks.
- `scripts/`: cloud artifacts, diagnostics, and model packaging.
- `docs/`: measured results, protocols, and research notes.

## Attribution and permissions

This project uses ideas from [Image-Adaptive 3D LUTs](https://github.com/HuiZeng/Image-Adaptive-3DLUT), [DeepLPF](https://github.com/sjmoran/deeplpf-image-enhancement), and related local-enhancement research.

No code license is granted at present. The [MIT–Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/) and other external materials retain their own terms. The examples above are included for illustration; full datasets, training logs, credentials, and model binaries are excluded from Git.
