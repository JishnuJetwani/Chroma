# Modal training

This workflow trains the original **2,748-scene** experiment on one H100. The [full FiveK workflow](full-fivek-training.md) extends its cache to 5,000 scenes.

## Prerequisites

Complete the JPEG collection, preview split, and additional-editor steps in the [data guide](data.md). These create `data/fivek/manifest.jsonl` and `data/fivek_multistyle/manifest.jsonl`, which the uploader reads.

```bash
uv sync --locked --extra app --extra data --extra dev --extra cloud
uv run --no-sync modal setup
uv run --no-sync modal run modal_train.py --preflight-only
```

The frozen [scene registry](../configs/splits/cloud_full_7030.json) specifies 1,924 training and 824 test scenes, with 1,958/838 editor pairs. Expert C supplies every scene; A/B each add 17 training and seven test targets. Expected manifest SHA256: `ddb01f569477fae9be510830fb7d154bb7d99ac2bc7c97393215a0106f191a3b`.

## Prepare, train, and fetch

```bash
uv run --no-sync python scripts/upload_modal_data.py --raw
uv run --no-sync modal run --detach modal_train.py \
  --prepare-only --cache cloud_full_7030_io32
```

After CPU preparation finishes, choose a new run ID and launch training:

```bash
uv run --no-sync python scripts/launch_modal_run.py jpeg-7030-run1
uv run --no-sync python scripts/fetch_modal_run.py jpeg-7030-run1
uv run --no-sync python scripts/report_modal_run.py runs/jpeg-7030-run1
```

Run the fetch/report commands after training completes. The fetched `runs/jpeg-7030-run1/joint/model.pt` is the portable checkpoint. Resumable optimizer states stay on the Modal run volume. Add `--resume` to the launch command to continue an interrupted run.

## Protocol

Training starts from fresh weights and follows 10 global, 5 synthetic-repair, and 10 joint epochs. Every training pair is used. Checkpoints follow the fixed schedule, with the test cache opened after training.

The runner benchmarks batch two with accumulation four against batch eight with accumulation one, and optionally CNN-only BF16. It selects runtime settings using training data, then reseeds the actual run. The reported experiment selected batch eight and FP32. LUT and local rendering remain FP32 in every configuration.

CPU preparation verifies source files in parallel and packs exact loader pixels into checksummed uint8 shards. The H100 keeps the training cache in device memory and executes all stages in one allocation. Epoch checkpoints support recovery within a cumulative one-hour budget; the Modal timeout is 65 minutes.

## Evaluation and dependencies

The final global and joint models are scored on the same 824 Expert C photos using PSNR, SSIM, CIE76, and clipping. Evaluation uses a custom 70/30 split and valid content on a padded 256-pixel canvas. The joint model has additional repair/joint updates. See the [results](reports/jpeg.md) for measured runtime and costs.

Cloud images install the hashed dependencies in `requirements-cloud.txt`. Regenerate them with:

```bash
uv export --locked --no-dev --no-emit-project --no-header \
  --output-file requirements-cloud.txt
```
