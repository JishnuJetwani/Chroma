# Complete FiveK JPEG experiment

3,500 training / 1,500 held-out scenes. Existing assignments were preserved; only previously missing scenes were partitioned.

Training uses all 3,500 Expert C pairs plus the unchanged 17 A and 17 B examples. Fixed curriculum: 10 global, 5 repair, 10 joint epochs; batch eight, FP32, seed 20260918. Final checkpoints are selected by schedule, with no test-based tuning.

## Same-image Expert C comparison

| Output | PSNR ↑ | SSIM ↑ | CIE76 color error ↓ |
|---|---:|---:|---:|
| Input | 18.414 dB | 0.7924 | 16.200 |
| Previous | 24.127 dB | 0.8987 | 8.224 |
| Global | 22.208 dB | 0.8715 | 9.940 |
| Joint | 24.446 dB | 0.9038 | 7.983 |

Previous is the existing 2,748-scene experiment's final local model; global and joint are the newly trained full-data models. All rows above score the identical 1,500 held-out photos.

## Original and additional holdouts

| Cohort | Photos | Previous PSNR | New joint PSNR | Paired improvement (95% CI) |
|---|---:|---:|---:|---:|
| All | 1500 | 24.127 | 24.446 | +0.319 [+0.233, +0.400] dB |
| Original | 824 | 23.981 | 24.320 | +0.340 [+0.234, +0.445] dB |
| Additional | 676 | 24.305 | 24.598 | +0.293 [+0.161, +0.417] dB |

## Runtime and interpretation

Recorded training-function time including cache loading and all three evaluations: 4.70 minutes on NVIDIA H100 80GB HBM3. Estimated function compute: $0.335; this is not a bill and excludes CPU preparation, image building, startup/idle, storage, and credits.

Newly clipped highlight pixels change by -0.063 percentage points versus the previous model (paired 95% CI [-0.089, -0.040]). This is an artifact measure, not an aesthetic preference result.
Newly clipped shadow pixels change by +0.853 percentage points versus the previous model (paired 95% CI [+0.725, +0.995]). This is an artifact measure, not an aesthetic preference result.

- Same epochs but more optimizer updates than the subset run.
- Custom 70/30 split and padded 256px metrics, not a published protocol.
- Original holdout has been inspected before; additional cohort was newly reserved.
- Single training seed; uncertainty is across test scenes, not training runs.

The original and added caches retain their image checksums and terms. The combined manifest, shard references, and scene registry are frozen beside the run artifacts. [Machine-readable summary](full_fivek_summary.json).

## Checkpoint and verification

The final local checkpoint is `runs/fivek-5000-7030-20261107/joint/model.pt`, SHA256 `90781d57ff40e0cb1416bbde97b0179f9409340743f44f4da0e5dc8a6d00693c`. Weights remain separate from Git.

Re-evaluating the previous checkpoint reproduced every Expert C PSNR value on the original 824 photos exactly. Native 720 × 480 exports of the flower and Paris examples passed a local smoke check; zero-strength rendering was an exact bypass, and source files stayed unchanged. This is a two-photo functional check, not a broad visual preference study.

See the [training protocol](../full-fivek-training.md) and [frozen split registry](../../configs/splits/fivek_5000_7030.json). Compute estimates use the [Modal resource rates](https://modal.com/pricing); the account invoice was not inspected.
