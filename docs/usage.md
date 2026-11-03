# Training and research workflow

A local photo grader that learns an overall color treatment and spatial exposure/contrast adjustments. A small CNN analyzes a preview; a differentiable renderer applies its recipe at the original resolution.

This is a research implementation. The local models and experiment tooling work; a pilot checkpoint is not evidence of expert-level editing, a novel algorithm, or state-of-the-art performance. See [research notes](research.md) and [measured results](results.md) for what has actually been tested.

## Setup

Python 3.11 and PyTorch 2.8. Apple Silicon uses MPS; CPU and CUDA are selectable. No custom GPU extensions are required.

```bash
uv sync --extra app --extra dev --extra data --extra raw
uv run photograder doctor
```

Without `uv`, create a Python 3.11 virtual environment and install `pip install -e '.[app,dev,data,raw]'`.
Original expert TIFF preparation also requires the LittleCMS2 shared library
(`brew install little-cms2` on macOS, `liblcms2-2` on Debian).

For optional Modal cloud tooling:

```bash
uv sync --extra app --extra dev --extra data --extra raw --extra cloud
uv run --no-sync --extra cloud modal setup
uv run --no-sync --extra cloud modal profile list
```

The `cloud` extra installs the Modal Python SDK and CLI; the lockfile currently pins version 1.5.5. Complete the browser sign-in opened by `modal setup` to connect your account. Installation and authentication do not launch a GPU job or upload photos. The optional [Modal training workflow](modal_training.md) prepares the public dataset on CPU, benchmarks one H100, and runs a bounded fresh 70/30 curriculum. The regular training commands below run locally.

## Use the app

The launcher now prefers the separate RAW model at
`runs/raw-pilot-20261030/shared/model.pt` when it exists. The **RAW** tab accepts
camera files such as DNG, RAF, NEF, ARW and CR2, subject to decoder support.
Press **Develop RAW**, compare neutral development with the learned grade,
and adjust grade strength or exposure. Exposure changes happen before display
tone mapping; highlight values above one remain available. Zero grade means
neutral RAW development, not a camera JPEG. Sliders reuse the analyzed recipe.
Export a new PNG/JPEG or a genuine 16-bit sRGB TIFF at native cropped dimensions.
The original sensor file is preserved. The **JPEG / PNG** tab uses the existing
JPEG model separately.

```bash
.venv/bin/python -m photograder app --raw-checkpoint runs/raw-pilot-20261030/shared/model.pt
```

RAW training uses original FiveK DNGs and 16-bit ProPhoto expert targets, with
float scene-linear caches and explicit ICC conversion. It does not treat an
inverse-gamma JPEG as RAW. See the [decoder research](raw-pipeline-research.md),
[architecture](raw-model-research.md), and [source preparation](raw-data-research.md).
The first RAW run is an exploratory pilot, not the previous full-data JPEG benchmark. See the [completed RAW results and known weaknesses](reports/raw.md).

```bash
# Generate a source plan only (HEAD requests; no source photo download).
.venv/bin/python -m photograder.raw_acquisition --probe-sizes
# Prepare public originals on CPU in Modal; retain only compact float pairs.
.venv/bin/modal run --detach modal_raw_train.py --prepare-only
# Four equal-update arms, preloaded data, one H100, explicit time cap.
.venv/bin/modal run --detach modal_raw_train.py --steps 800 --budget-seconds 900
```

Each RAW run records source versions, checksums, split registry, code snapshots,
training losses, per-scene validation metrics and exclusion reasons. Its inner
validation scenes come only from the old outer training pool. The old 824-scene
outer test remains excluded from development. Missing white-balance data,
unsupported sensors, fractional/unverified crops or incompatible geometry are
reported as exclusions rather than silently guessed. RAW still cannot restore
detail the sensor never captured; Fujifilm RAF quality needs camera-specific validation.

For the JPEG path, the full-data model is at `runs/full-7030-20261012/joint/model.pt`, trained from scratch on 1,924 scenes and evaluated on 824 held-out scenes. It offers FiveK Experts C, A and B, but A/B each have only 17 training scenes, so preset learning remains limited. See the [full-data results](reports/jpeg.md). Earlier pilot checkpoints remain available as fallbacks.

```bash
uv run --extra app photograder app --checkpoint runs/full-7030-20261012/joint/model.pt
```

Open http://127.0.0.1:7860. Choose a JPEG/PNG, press **Auto Grade**, compare before/after, and export a new file. Strength sliders re-render the cached recipe without another CNN call. The UI offers only styles present in the checkpoint, disables public sharing, and keeps photos local. Nonopaque transparent PNGs must be flattened first. Exports are 8-bit sRGB at the original dimensions; originals are never overwritten.

For washed-out photos with strong sunlight, the local model also offers **Treatment → Backlight recovery (experimental)**. It smooths the predicted color table, reduces aggressive local edits, and protects source highlights while applying a gentle foreground tone correction. Standard grade remains the default: the experimental mode helped the supplied development photo but reduced average target fidelity on a varied 36-image training-set check. See the [five-direction investigation, experiments, and next training plan](backlight-next-steps.md). The existing held-out benchmark describes Standard grade, not this treatment.

For one file:

```bash
uv run photograder grade photo.jpg --checkpoint runs/local_pilot/best.pt \
  --output exports/photo-graded.png --recipe exports/photo-recipe.pt
```

The `.pt` recipe stores both the LUT and spatial maps. A `.cube` LUT alone cannot reproduce the local adjustments.
Add `--treatment backlight` to the `grade` command to reproduce the experimental treatment. Its saved recipe includes the smoothing result and tone-policy metadata used for export.

## Data

The downloader reads the remote ZIP directory and fetches only selected FiveK JPEG entries with HTTP range requests, validating their ZIP CRCs. It refuses servers that substitute a full-archive response. It also verifies available storage and writes SHA256 manifests and provenance.

```bash
# Small real-photo pipeline check: 16 train, 4 validation, 4 test.
uv run photograder acquire --output data/fivek_pilot

# Full supplied split: 2025 train, 225 validation, 498 published test.
uv run photograder acquire --full --output data/fivek --workers 8
uv run photograder verify-data data/fivek/manifest.jsonl
```

The source release exposes 2250 training IDs and 498 test IDs. Validation is a deterministic held-out subset of those training IDs, not the unrelated additional source photos. This is our documented split and must not be confused with other FiveK protocols.

The main downloader acquires input RGB and Expert C targets. A separate bounded importer has also prepared 24 matched scenes with Expert C/A/B renditions in `data/fivek_multistyle/manifest.jsonl`. Original 16-bit A/B TIFFs pass through native ColorSync before conversion to 8-bit sRGB. See [verified data and acquisition commands](data.md). The dataset's research terms are saved with the download. Never commit dataset images to the repository. S24 targets still require acquisition; the model does not invent human style labels.

`photograder.data.prepare_pairs` imports your existing paired files. The S24 importer preserves 16-bit RAW RGB during color conversion, applies camera white balance/CCM, produces a fixed sRGB source, and excludes privacy blur masks. See `python -m photograder.acquisition --help` for importer options. The training pipeline is JPEG/PNG-first; this is not a RAW reconstruction benchmark.

## Train and compare

```bash
uv run photograder benchmark --config configs/local_pilot.json --steps 200
uv run photograder train --config configs/fixed_pilot.json
uv run photograder train --config configs/global_pilot.json
uv run photograder train --config configs/repair_pilot.json \
  --initialize runs/global_pilot/best.pt
uv run photograder train --config configs/local_pilot.json \
  --initialize runs/repair_pilot/best.pt
uv run photograder train --config configs/local_no_preservation_pilot.json \
  --initialize runs/repair_pilot/best.pt
```

Every run writes resolved configuration, environment, history, summary, and `best.pt`/`last.pt`. Existing experiments cannot be silently overwritten. `--initialize` starts a new experiment from compatible weights; `--resume` continues an existing experiment with the same data and training protocol. Use a new output directory to change the method or dataset.

Style transfer matches embedding rows by editor name, including when adding new styles. New checkpoints record an optimizer-boundary cursor so `--resume` continues remaining batches after a partial epoch. Legacy checkpoints without that cursor retain their next-epoch behavior. Corruption-generator changes require a fresh experiment.

CLI overrides include `--manifest`, `--limit`, `--val-limit`, `--image-size`, `--epochs`, `--max-steps`, `--seed`, and `--device`. The first pilots use limited subsets; remove limits and extend the budget only after inspecting results.

```bash
uv run photograder experiments configs/matrix.json             # dry-run
uv run photograder experiments configs/matrix.json --execute  # sequential, local
uv run photograder evaluate --checkpoint runs/local_pilot/best.pt \
  --manifest data/fivek/manifest.jsonl --split test \
  --output runs/local_pilot/test --controls --save-images
```

The exact 256-pixel preview pilot is replayable through `configs/matrix_preview.json`, which uses fresh output directories. The completed global continuation matches the spatial curriculum's allocated 400 optimizer updates. Reproduce it with `configs/global_matched_pilot.json --initialize runs/global_pilot/best.pt` (160 inherited plus 240 new updates). The initial 160-update global row alone has an unequal budget; the completed comparison is in the results report. The completed three-editor pilot uses `configs/multistyle_pilot.json --initialize runs/local_pilot/best.pt`; its 24 scenes are too few for a style-quality claim.

```bash
uv run photograder benchmark-inference your-photo.jpg \
  --checkpoint runs/local_pilot/best.pt --output runs/export_benchmark
```

Evaluation reports paired PSNR/SSIM/Lab CIE76, clipping, and optional controlled local-repair metrics with scene-level bootstrap intervals. It separates errors inside the affected region from drift outside it. It does not call a metric a human aesthetic score. Evaluation images are ordered original / model output / expert target.

## Architecture

| Architecture | Parameters, one style | Purpose |
|---|---:|---|
| `fixed` | 107,811 | One learned LUT per style |
| `global` | 1,447,248 | Image-adaptive mixture of eight LUTs |
| `local` | 1,711,026 | Shared encoder plus local exposure/contrast maps |
| `cascade` | 3,142,004 | Independent sequential specialists; larger capacity |
| `unet` | 847,411 | Preview RGB-residual alternative |
| `bilateral` | 597,616 | Small spatial affine-grid alternative |

The alternatives are internal implementations, not reproductions of named papers. Parameter counts are reported rather than assuming these are capacity-matched. `tonal_specialists`, `order`, and `upsampling` configuration options expose the specialist, operation-order, and guided-upsampling ablations.

The default encoder uses 32/64/128/128 channels, a 16-dimensional style embedding, 256-pixel previews, 64-pixel local maps, and eight 33³ RGB tables. Exposure is bounded to ±1 stop and contrast to 0.5–2.0. Local heads start at identity. Rendering uses gather-based LUT interpolation, guided spatial upsampling, a guided log-luminance base/detail decomposition, and a single final clamp. It cannot reconstruct texture absent from fully clipped highlights.

## Tests

```bash
uv run --extra dev pytest -q
```

Tests include finite-difference LUT gradients, CPU/MPS parity, tile/full-render parity, identity and adjustment direction, scene leakage, archive safety/CRC handling, color management, deterministic local faults, masked metrics, checkpoint roundtrips, training loss reduction, cached previews, and source preservation.

## Research artifacts

- [Research manuscript draft and related work](research.md)
- [Pilot experimental paper](paper.md)
- [Experiment protocols and blind preference studies](experiments.md)
- [Measured results and limitations](results.md)

Cloud execution is explicit through `modal_train.py`; the local app never uploads photos. The Modal workflow uploads only manifest-listed public training pairs and launches one bounded GPU run when invoked. No automatic experiment sweep is enabled.
