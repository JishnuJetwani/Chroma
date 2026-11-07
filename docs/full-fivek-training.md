# Complete FiveK JPEG training

This experiment expands the existing 2,748-scene experiment to every one of the 5,000 matching 480p JPEG input/Expert C pairs in the source archive. It preserves the old scene assignments and preprocessing.

| Partition | Existing scenes | Additional scenes | Total |
|---|---:|---:|---:|
| Training | 1,924 | 1,576 | 3,500 |
| Test | 824 | 676 | 1,500 |

The same 17 training and seven test scenes with A/B edits remain in their original partitions. There are 3,534 training pairs and 1,514 test pairs across the three styles; the primary evaluation uses only Expert C. Every version of a scene remains in one partition.

The [frozen registry](../configs/splits/fivek_5000_7030.json) records every assignment, cohort, and source image hash. Its manifest SHA256 is `1d502cf6876a4161867bbe900a7bb4403d0f2d3f2ca9bbb7241f3a0271dd9bc2`.

## Acquisition and cache

`modal_fivek_full.py` uses a CPU-only Modal function to inventory the archive and download only the 2,252 missing input/target pairs. It coalesces bounded HTTP ranges, verifies ZIP CRC32/size and SHA256, and checks input/target geometry during cache preparation. It uses the same source JPEG rendering and the existing EXIF/ICC-aware 256-pixel dataset loader. No new RAW rendering is introduced.

The previously uploaded `cloud_full_7030_io32` cache is a prerequisite. Its original manifest hash is checked before training. The expanded index references its immutable tensor shards alongside the new shards rather than duplicating or rewriting them. All source checksums, download provenance, research terms, and cohort membership are retained on the data volume.

CPU preparation saves `/data/fivek_5000_7030_v1/protocol.json` and returns that protocol locally. The GPU runner requires the independently frozen manifest hash and exact scene/pair counts. Test tensors are loaded only after the entire training schedule finishes.

## Training and evaluation

- Fresh initialization, then 10 global epochs, 5 synthetic-repair epochs, and 10 joint epochs.
- Same architecture, losses, seed 20260918, AdamW learning rate 0.0002, and weight decay 0.0001 as the subset experiment.
- Batch size eight, accumulation one, FP32 throughout. The previous run's runtime choice is fixed; a short training-only throughput check gates the run.
- Every training pair is used. There is no validation set or test-based checkpoint selection.
- One H100, a 20-minute cooperative function budget, a 25-minute outer timeout, and epoch checkpoints on the run volume.
- Final global and local models, plus the previous local model, are evaluated on the identical expanded holdout.
- Report all 1,500 Expert C scenes, the original 824, and the newly added 676 separately, including paired scene-bootstrap intervals.

Keeping the epoch schedule means more optimizer updates on the larger dataset. This experiment tests the expanded training recipe; it does not isolate data volume from additional optimization. The old holdout has been inspected before, so the newly reserved cohort is reported separately. A custom 70/30 split and padded 256-pixel scores are not directly comparable with published FiveK protocols.

## Commands

```bash
uv sync --locked --extra cloud --extra dev
# CPU preparation; creates the frozen local protocol.
uv run --no-sync modal run --detach modal_fivek_full.py
# Train only after preparation succeeds.
uv run --no-sync modal run --detach modal_fivek_full.py --train
uv run --no-sync python scripts/fetch_modal_run.py fivek-5000-7030-20261107
```

Download the frozen registry from the data volume before generating the comparison report:

```bash
uv run --no-sync modal volume get selective-photograder-data-v1 \
  fivek_5000_7030_v1/split_registry.json \
  runs/fivek_full_20261107/split_registry.json
uv run --no-sync python scripts/report_fivek_expansion.py \
  runs/fivek-5000-7030-20261107 \
  runs/fivek_full_20261107/split_registry.json
```

Existing caches and completed runs are not overwritten. Use `--train --resume` only for an interrupted run after checking its status; a completed run is rejected. Model weights and full reports remain local/cloud artifacts unless separately packaged for publication.
