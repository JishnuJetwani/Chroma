# Experiments

Use [local training](usage.md#train-a-small-model) for a first run and [full FiveK training](full-fivek-training.md) for the latest protocol. This guide covers architecture comparisons and the original development pilots.

## Configurations

| Configuration | Comparison |
|---|---|
| `fixed_pilot.json` | One LUT per style |
| `global_pilot.json` | Image-adaptive mixture of LUTs |
| `repair_pilot.json` | Local exposure/contrast repair |
| `local_pilot.json` | Joint global and local grading |
| `local_no_preservation_pilot.json` | Joint model with outside-consistency loss disabled |
| `local_bilinear_pilot.json` | Bilinear map upsampling |
| `local_specialists_pilot.json` | Separate tonal specialists |
| `cascade_pilot.json` | Sequential networks |
| `unet_pilot.json` | Preview RGB residual |
| `bilateral_pilot.json` | Local affine grid |

These files live in [configs](../configs/). The default matrix trains fixed → global → repair → local → local without preservation. Both joint variants initialize from the same repair checkpoint. Matrix execution uses the device in each config; the supplied local pilots select MPS.

```bash
uv run --no-sync photograder experiments configs/matrix.json
uv run --no-sync photograder experiments configs/matrix.json --execute --max-runs 1
```

The first command previews the plan. Prepare the manifests using the [data guide](data.md) before executing it.

## Replay the measured preview pilots

The [preview registry](../configs/splits/fivek_preview.json) fixes 128 training, 32 validation, and 32 test scenes. Rebuild its manifest, then run:

```bash
uv run --no-sync photograder experiments configs/matrix_preview.json --execute
```

This uses 256-pixel analysis/rendering, seed 42, patience eight, and fresh outputs under `runs/preview_reproduction`. Fixed/global allocate 160 updates; each local curriculum allocates 160 global + 80 repair + 160 joint updates. Validation L1 selects stage checkpoints. Hardware nondeterminism can change the final numbers.

A separate global continuation adds 240 updates to the initial 160-update model, matching the 400 allocated updates of the local curriculum. The [pilot results](results.md) record the selected checkpoint lineages, parameter counts, and paired differences.

## Evaluate and compare

For example, evaluate the checkpoint produced by the preview matrix:

```bash
uv run --no-sync photograder evaluate \
  --checkpoint runs/preview_reproduction/local/best.pt \
  --manifest data/fivek/manifest.preview.jsonl --split test \
  --image-size 256 --device cpu --seed 42 \
  --output runs/preview_reproduction/local/test --controls --save-images
```

Evaluate the other models with the same options, then compare their report directories:

```bash
uv run --no-sync python -m photograder.experiments compare \
  runs/preview_reproduction/global/test \
  runs/preview_reproduction/local/test \
  runs/preview_reproduction/local_no_preservation/test \
  --output-dir runs/preview_reproduction/comparison
```

The first report is the reference. The comparison helper requires matching manifests, scene IDs, styles, resolution, device, seed, and metric definitions. It reports candidate-minus-reference deltas with paired scene-bootstrap intervals.

Metrics include encoded-sRGB PSNR, SSIM with Gaussian windows and fully valid support, D65 Lab CIE76, and introduced clipping. Controlled faults report target error inside the affected region and output drift outside it. Outside drift compares grades of the clean and corrupted inputs at the same style. Saved panels show input / model / expert target.

Match data, initialization, and update budgets for architecture comparisons; also record wall time and parameter counts. Use validation for selection, and reserve the test set for the final comparison.

## Blind comparisons

Create `comparisons.json` with records containing `scene_id`, unique `image_id`, `input_path`, `a_path` for the baseline, and `b_path` for the candidate. Use matching dimensions and color encoding.

```bash
uv run --no-sync python -m photograder.experiments blind \
  comparisons.json evaluation/blind --seed 42
uv run --no-sync python -m photograder.experiments votes \
  evaluation/blind/votes.csv evaluation/blind/private_key.csv \
  --candidate candidate --output evaluation/blind/results.json
```

Give raters `sheets/` and a blank `votes.csv`; retain the private key separately. Each rater enters an anonymous ID and A/B/tie answers. The analyzer counts ties and unanswered questions, rejects duplicate answers, and bootstraps scenes after averaging ratings within images.

## External references and resume

For external checkpoints, record the source revision, weight hash, training split, rendering, and any scene overlap. The [DeepLPF pilot comparison](results.md#external-deeplpf-reference) used an author checkpoint with a larger training budget.

`--resume` checks the data hash, style registry, architecture, losses, and corruption version. Current checkpoints resume at a saved optimizer boundary; legacy checkpoints resume at the next epoch. Use a new run when changing the protocol.
