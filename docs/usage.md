# Usage

## Setup

Install Python 3.11 and [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked --extra app --extra data --extra dev
uv run --no-sync photograder doctor
```

CPU, Apple MPS, and NVIDIA CUDA are supported. Add `--device cpu`, `mps`, or `cuda` to select a device explicitly.

## Grade a photo

Obtain weights using the [model card](model-card.md), then save them as `checkpoints/jpeg.pt`:

```bash
uv run --no-sync photograder app --checkpoint checkpoints/jpeg.pt
```

Open **http://127.0.0.1:7860**, upload a JPEG or PNG, and press **Auto Grade**. Strength sliders reuse the cached recipe. Exports are new 8-bit sRGB files at the original dimensions. Flatten transparent PNGs before uploading.

For the command line, replace `photo.jpg` with your image:

```bash
uv run --no-sync photograder grade photo.jpg \
  --checkpoint checkpoints/jpeg.pt \
  --output exports/photo-graded.png --recipe exports/photo-recipe.pt
```

The saved recipe contains the LUT and spatial maps. Add `--treatment backlight` to try the optional [backlight treatment](backlight-next-steps.md).

## Train a small model

This sequence downloads 24 FiveK pairs, then trains global, repair, and joint stages. It creates the manifests and checkpoints referenced by later commands.

```bash
uv run --no-sync photograder acquire --output data/fivek_pilot
uv run --no-sync photograder verify-data data/fivek_pilot/manifest.jsonl
uv run --no-sync photograder train --config configs/global_pilot.json \
  --manifest data/fivek_pilot/manifest.jsonl --device auto
uv run --no-sync photograder train --config configs/repair_pilot.json \
  --manifest data/fivek_pilot/manifest.jsonl --device auto \
  --initialize runs/global_pilot/best.pt
uv run --no-sync photograder train --config configs/local_pilot.json \
  --manifest data/fivek_pilot/manifest.jsonl --device auto \
  --initialize runs/repair_pilot/best.pt
uv run --no-sync photograder evaluate \
  --checkpoint runs/local_pilot/best.pt \
  --manifest data/fivek_pilot/manifest.jsonl --split test \
  --image-size 256 --output runs/local_pilot/evaluation --save-images
```

Each run saves configuration, environment, training history, and `best.pt`/`last.pt`. Use a fresh `--output` directory for another experiment. `--initialize` transfers weights into a new run; `--resume` continues an existing run with its saved data and settings.

For larger runs, follow [data preparation](data.md), the [experiment guide](experiments.md), or [full FiveK training](full-fivek-training.md). Configuration overrides include `--manifest`, `--image-size`, `--epochs`, `--max-steps`, `--batch-size`, and `--device`.

## RAW developer

Install the RAW dependencies and place a compatible RAW checkpoint at `checkpoints/raw.pt`:

```bash
uv sync --locked --extra app --extra data --extra dev --extra raw
uv run --no-sync photograder app \
  --checkpoint checkpoints/jpeg.pt --raw-checkpoint checkpoints/raw.pt
```

The RAW tab provides exposure and grade controls, a neutral-development comparison, and PNG/JPEG or 16-bit sRGB TIFF export. Camera support depends on LibRaw. The [RAW pilot report](reports/raw.md) covers the trained model and tested files.

Preparing original expert TIFF targets also requires LittleCMS2: `brew install little-cms2` on macOS or the `liblcms2-2` package on Debian. Follow the [RAW data guide](raw-data-research.md) for acquisition and training.

## Performance and development

```bash
uv run --no-sync photograder benchmark-inference photo.jpg \
  --checkpoint checkpoints/jpeg.pt --output runs/export_benchmark
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

Inference benchmarking records analysis, cached preview, and native export timings. Tests cover rendering gradients, CPU/MPS parity, tiled output, color management, split integrity, checkpoint loading, and exports.
