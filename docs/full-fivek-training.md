# Full FiveK training

The full-data experiment uses all 5,000 matching 480p JPEG input/Expert C pairs, preserving the earlier split and adding the remaining scenes:

| Partition | Existing | Added | Total |
|---|---:|---:|---:|
| Training | 1,924 | 1,576 | 3,500 |
| Test | 824 | 676 | 1,500 |

A/B edits retain their original assignments, bringing the totals to 3,534 training and 1,514 test pairs. The primary evaluation uses Expert C. The [registry](../configs/splits/fivek_5000_7030.json) records scene assignments, source hashes, and cohorts. Manifest SHA256: `1d502cf6876a4161867bbe900a7bb4403d0f2d3f2ca9bbb7241f3a0271dd9bc2`.

## Cloud prerequisites

The checked-in runner extends the earlier Modal experiment. It requires these existing volume artifacts:

| Volume | Required artifact |
|---|---|
| `selective-photograder-data-v1` | `cloud_full_7030_io32` cache from the [subset workflow](modal_training.md) |
| `selective-photograder-runs-v1` | `full-7030-20260918/joint/model.pt`, SHA256 `1e60aaab4c95a9c3755a6d6f314c4c531b355d984dc8edab5c8f776c5e8d657a` |

The second artifact is the pinned comparison model. A new Modal account needs these artifacts provisioned before using this runner. For an initial training run, use the [local guide](usage.md#train-a-small-model) or [subset workflow](modal_training.md).

## Acquire and cache

CPU preparation inventories the archive and downloads the 2,252 missing pairs through bounded HTTP ranges. It checks CRC32, sizes, SHA256, and geometry, then applies the same EXIF/ICC-aware 256-pixel loader as the existing cache. The combined index references both sets of immutable shards.

```bash
uv sync --locked --extra cloud --extra dev
uv run --no-sync modal run --detach modal_fivek_full.py
```

Preparation writes the local `runs/fivek_full_20261107/protocol.json` and the combined cloud cache. Once it finishes, train with a fresh run ID:

```bash
uv run --no-sync modal run --detach modal_fivek_full.py \
  --train --run-id fivek-full-run1
```

## Training settings

- Fresh initialization; 10 global, 5 repair, and 10 joint epochs.
- AdamW, learning rate 0.0002, weight decay 0.0001, seed 20260918.
- Batch eight, accumulation one, FP32; 11,050 optimizer updates.
- Final scheduled checkpoints; all 3,500 training scenes used.
- One H100, 20-minute cooperative budget, 25-minute function timeout.

The previous run used the same epochs with fewer examples and 6,125 updates. The comparison therefore includes both additional data and additional optimization.

## Fetch and evaluate

After training completes:

```bash
uv run --no-sync python scripts/fetch_modal_run.py fivek-full-run1
uv run --no-sync modal volume get selective-photograder-data-v1 \
  fivek_5000_7030_v1/split_registry.json \
  runs/fivek_full_20261107/split_registry.json
uv run --no-sync python scripts/report_fivek_expansion.py \
  runs/fivek-full-run1 runs/fivek_full_20261107/split_registry.json
```

The final global, final local, and previous local checkpoints are scored on identical photos. Reports separate the full 1,500-photo holdout, original 824, and added 676, with paired scene-bootstrap intervals. The resulting portable checkpoint is `runs/fivek-full-run1/joint/model.pt`.

See the [completed experiment](reports/full-fivek.md) for results. This protocol uses a custom 70/30 split and padded 256-pixel evaluation.
