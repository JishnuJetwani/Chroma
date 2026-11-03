# Selective Grade

Automatic photo grading with a small CNN, learned color tables, and local exposure and contrast adjustments. Analyze a photo once, adjust the strength, and export at its original resolution. The app runs locally.

The main JPEG model has **1.71 million parameters**. On 824 held-out FiveK photos, it reached **23.98 dB PSNR / 0.8958 SSIM**, compared with **18.34 dB / 0.7906** for the unedited inputs. These measure agreement with Expert C, not aesthetic quality. [Benchmark and limitations](docs/reports/jpeg.md).

## Try it

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). CPU, Apple MPS, and NVIDIA CUDA are supported; no custom GPU extensions are needed.

```bash
uv sync --locked --extra app
uv run --no-sync photograder doctor
```

Place the trained JPEG checkpoint at `checkpoints/jpeg.pt`, then run:

```bash
uv run --no-sync photograder app --checkpoint checkpoints/jpeg.pt
```

Open **http://127.0.0.1:7860**, upload a JPEG or PNG, and press **Auto Grade**. The overall-grade and local-correction sliders reuse the cached prediction. Export creates a new file and preserves the original.

**Weights are separate from the source repository.** There is no hosted download yet. The existing local checkpoint at `runs/full-7030-20261012/joint/model.pt` also works with `--checkpoint`. See the [model card](docs/model-card.md) for its checksum, training data, and limitations, or train your own using the commands below. On macOS, `run_app.command` discovers either location.

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

The model predicts **editing operations**, not replacement image pixels. Exposure is bounded to ±1 stop, and contrast multiplies local log-luminance detail by 0.5–2.0. A learned style embedding conditions both branches. Setting both strengths to zero is an exact bypass.

The code also includes a fixed LUT baseline, cascaded specialists, an RGB-residual U-Net, and a bilateral-grid alternative. These are experimental comparisons; their presence does not imply they beat the main model.

## Data and training

The current JPEG run uses **2,748 unique FiveK scenes**, not the entire 5,000-photo dataset: 1,924 for training and 824 for testing. The main target is Expert C. Editors A/B each have only 17 training scenes, so those extra styles are exploratory. S24 has an importer but was not used in the reported runs.

For a small pipeline check:

```bash
uv sync --locked --extra app --extra data --extra dev
uv run --no-sync photograder acquire --output data/fivek_pilot
uv run --no-sync photograder train --config configs/global_pilot.json \
  --manifest data/fivek_pilot/manifest.jsonl --device auto
uv run --no-sync photograder train --config configs/repair_pilot.json \
  --manifest data/fivek_pilot/manifest.jsonl --device auto \
  --initialize runs/global_pilot/best.pt
uv run --no-sync photograder train --config configs/local_pilot.json \
  --manifest data/fivek_pilot/manifest.jsonl --device auto \
  --initialize runs/repair_pilot/best.pt
```

This small run checks the pipeline; it does not reproduce the main benchmark. All variants of a source photo stay in the same split. Training refuses to overwrite an existing run. The [training guide](docs/usage.md), [data notes](docs/data.md), and [optional Modal workflow](docs/modal_training.md) cover larger runs and evaluation.

## Results

| Output on Expert C holdout | PSNR ↑ | SSIM ↑ | Color error, CIE76 ↓ |
|---|---:|---:|---:|
| Original input | 18.34 dB | 0.7906 | 16.30 |
| Global LUT model | 21.97 dB | 0.8676 | 10.25 |
| Global + local model | **23.98 dB** | **0.8958** | **8.35** |

The global and final models have different total training budgets, so this comparison does not isolate the effect of the local architecture. The custom split and 256-pixel evaluation also differ from other published FiveK protocols. [Detailed report](docs/reports/jpeg.md) · [Metrics and confidence intervals](docs/reports/jpeg-metrics.json).

Highlights already clipped in a JPEG cannot be recovered. Aggressive corrections can introduce clipping or banding. The experimental backlight treatment is optional and is **not** the model measured in the table.

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

This project uses established ideas from [Image-Adaptive 3D LUTs](https://github.com/HuiZeng/Image-Adaptive-3DLUT), [DeepLPF](https://github.com/sjmoran/deeplpf-image-enhancement), and related local-enhancement research. It is a personal research implementation, not a claim of a new state-of-the-art method. [Related work](docs/research.md).

No code license is granted at present. The [MIT–Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/) and other external materials retain their own terms. Dataset photos, private uploads, training logs, credentials, and model binaries are excluded from Git. The trained weights do not come with a commercial-use permission grant from this project.
