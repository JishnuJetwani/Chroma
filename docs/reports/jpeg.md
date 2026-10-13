# JPEG model benchmark

Recorded 2026-10-12 on **NVIDIA H100 80GB HBM3**. Fixed schedule: 10 global + 5 repair + 10 joint epochs.

1,924 training and 824 held-out scenes; 1,958/838 editor pairs. No test-based selection. All stages started from the same fresh curriculum, with no earlier pilot weights.

## Primary Expert C holdout (824 photos)

| Output | PSNR ↑ | SSIM ↑ | CIE76 color error ↓ |
|---|---:|---:|---:|
| Input | 18.339 dB | 0.7906 | 16.297 |
| Global | 21.974 dB | 0.8676 | 10.254 |
| Joint | 23.981 dB | 0.8958 | 8.346 |

Joint minus global PSNR: **+2.007 dB**, paired scene-bootstrap 95% CI [+1.748, +2.255].

Newly clipped highlight pixels increase by **0.227 percentage points** versus global-only (95% CI [0.181, 0.281]). This is a measurable tradeoff requiring visual inspection; clipping alone does not prove an unwanted edit.

SSIM is a structural similarity index, not a percentage of photographic quality or an accuracy score. The global and joint checkpoints have different total training budgets.

## Runtime and cost

Recorded GPU function work across attempts: 6.07 minutes. Estimated function compute: $0.432; this is not the account bill and excludes preparation, startup/idle, any unrecorded cancellation tail, storage and credit adjustments.
Selected microbatch 8, accumulation 1, CNN precision none; renderer FP32.

## Limits

The custom split differs from the published FiveK split and earlier pilot holdout, so do not directly compare their means. Metrics use padded 256px images with valid-content masks. No convergence claim is made after a fixed schedule. A/B each have only seven test scenes and 17 training scenes. This run does not establish real local defect detection, preservation-training causality, halo-free full-resolution rendering, or blind preference wins.


The portable checkpoint is documented in the model card (`../model-card.md`). Source images and local run logs are not included in the repository.
