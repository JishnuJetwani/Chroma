# Full-data Modal training

This experiment starts from scratch on a fresh, grouped 70/30 partition of all
currently acquired public pairs. Earlier pilot checkpoints must not initialize
it: they have seen some scenes in the new test partition.

## Frozen protocol

- Seed: `20260918`; scene assignment is deterministic SHA256 ranking.
- 2,748 distinct FiveK scenes: **1,924 train / 824 test**, rounded to whole scenes.
- 2,796 unique scene/editor pairs: **1,958 train / 838 test**. Duplicate C rows
  are removed. Crops, alternate edits and synthetic variants stay with their scene.
- Training styles: 1,924 C, 17 A, 17 B. Test: 824 C, seven A, seven B. Sampling
  follows observed rows; the scarce A/B targets are not oversampled. This is
  primarily a natural-grade experiment, not evidence of mature preset learning.
- No validation set, early stopping, or checkpoint selection using test images.
- Fixed curriculum: ten global epochs, five synthetic-repair epochs, ten joint
  epochs. Each stage uses every training pair. Final stage checkpoints are selected
  by the declared schedule. AdamW, learning rate 0.0002, weight decay 0.0001.
- Effective batch eight; benchmark microbatch two/accumulation four against eight/
  accumulation one, with optional CNN-only BF16. Changing the microbatch also
  changes the deterministic synthetic-example cadence, so these are not claimed
  bit-identical training trajectories. The optimizer update budget is preserved.
- Training and metric canvases are 256px with aspect-preserving padding. Full
  resolution export remains available, but these are not full-resolution metrics.
- This is a custom research split, not the published FiveK benchmark split.

The immutable registry is `configs/splits/cloud_full_7030.json`. Expected manifest
SHA256: `ddb01f569477fae9be510830fb7d154bb7d99ac2bc7c97393215a0106f191a3b`.
Raw files retain the original dataset terms and source checksums. Only acquired
public FiveK pairs are uploaded, not personal photos, app uploads, or the workspace.

## Efficient execution

Image dependencies are built before any GPU call. Since the Mac is near its disk
reserve, raw public pairs are uploaded once and converted on a CPU-only Modal
function into a compressed uint8 NPZ cache. Its exact pixels match the existing
dataset loader; each shard is checksummed. This temporary cloud preparation avoids
creating another large local cache.

Source-file verification uses 32 concurrent reads to tolerate cloud storage
latency; image processing remains capped at four CPU workers. Cache readers verify
up to eight shards concurrently before use, then decode one shard at a time.

The H100 loads the training uint8 cache once into device memory. Minibatches are
selected on-device and converted to FP32 only when used. The network uses scoped
BF16 only if it passes a training-image numerical check and improves measured
runtime; LUT interpolation, local tone calculations and output recipes stay FP32.
Scalar logging is batched, and the three dependent curriculum stages run in one
allocation. Extra GPUs are not provisioned for this small network.

Before fitting retained weights, benchmark 32 measured microbatches after eight
warmups for each stage and candidate. This includes the real loss, synthetic
corruptions, backward, clipping, and accumulated optimizer updates. Discard these
benchmark models. Select runtime settings using training data only, then reseed
and create fresh models for actual training. A conservative forecast gate avoids
starting a curriculum that clearly will not fit the runtime budget.

A cooperative one-hour budget covers cache loading, benchmarking, training and
evaluation; Modal also enforces a 65-minute outer timeout. Checkpoints are committed
after every epoch. Time stops preserve an optimizer-boundary resume cursor and do
not label partial training as complete. GPU spending estimates use actual elapsed
time; startup, CPU preparation, storage, idle tail, credits and billing adjustments
are accounted for separately. Account rates read on 2026-10-12 are $3.95/H100-hour,
$0.04730/physical-CPU-hour and $0.008/GiB-hour: approximately **$4.27/hour** for
one H100, four physical CPU cores and 16 GiB RAM.

## Commands

```bash
uv sync --extra app --extra data --extra dev --extra cloud
.venv/bin/modal setup
.venv/bin/modal run modal_train.py --preflight-only
.venv/bin/python scripts/upload_modal_data.py --raw
.venv/bin/modal run --detach modal_train.py --prepare-only --cache cloud_full_7030_io32
.venv/bin/python scripts/launch_modal_run.py full-7030-20261012
.venv/bin/python scripts/fetch_modal_run.py full-7030-20261012
.venv/bin/python scripts/report_modal_run.py runs/full-7030-20261012
```

Uploads and fresh preparations refuse to overwrite an existing dataset. Reuse the
prepared Volume for later runs; give each run a new ID. Do not repeat paid training
just to inspect results. Downloaded `joint/model.pt` contains the final weights and
metadata without duplicate optimizer states; full resumable states stay on the
run Volume. `best.pt` is an explicit alias for the fixed final checkpoint in this
protocol, not a claim that test or validation selected it.

The launcher uses a separate process session and writes a durable local log. This
prevents an interrupted chat/tool call from cancelling an attached Modal client.
If a job is confirmed stopped, `scripts/launch_modal_run.py RUN_ID --resume`
reuses completed phases and the measured runtime settings; partial phases resume
from their saved optimizer boundary. Resume verifies the data, schedule and all
learning/evaluation source hashes. It does not reset the cumulative recorded
runtime budget. Time between the final durable report and cancellation is not
included in that recorded runtime, so account billing remains authoritative.

## Evaluation

The test cache is first loaded after all 25 training epochs have completed. Score
the final global baseline and final joint model against the same 824 held-out
scenes, with input-only metrics as a reference. Report PSNR, SSIM, CIE76 color
difference and clipping, including per-style breakdowns and scene-bootstrap
uncertainty. Global versus joint has different total training budgets and does
not by itself establish the preservation method's novelty.

The run does not repeat the full synthetic-control or human-preference experiment.
Those remain separate research steps. Any architecture or loss changes informed
by this test evaluation would turn it into development data; a new final holdout
would then be needed.

Cloud images install the pinned, hashed transitive dependencies exported from `uv.lock` in `requirements-cloud.txt`. Regenerate with `uv export --locked --no-dev --no-emit-project --no-header --output-file requirements-cloud.txt`.
