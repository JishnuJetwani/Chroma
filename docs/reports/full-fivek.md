# Full FiveK results

The full-data JPEG model reaches **24.45 dB PSNR / 0.9038 SSIM** on 1,500 held-out Expert C photos. It improves PSNR by **0.319 dB** over the previous model on the same images.

## Protocol

The custom 70/30 split contains 3,500 training and 1,500 test scenes. Existing assignments are preserved; the additional scenes fill out each partition. Training includes 17 A and 17 B targets alongside the 3,500 C targets.

The fixed curriculum uses 10 global, 5 repair, and 10 joint epochs, batch eight, FP32, and seed 20260918. Final checkpoints are selected by schedule. Metrics use valid content on padded 256-pixel images.

## Same-image comparison

| Output | PSNR ↑ | SSIM ↑ | CIE76 color error ↓ |
|---|---:|---:|---:|
| Input | 18.414 dB | 0.7924 | 16.200 |
| Previous local model | 24.127 dB | 0.8987 | 8.224 |
| New global model | 22.208 dB | 0.8715 | 9.940 |
| New local model | **24.446 dB** | **0.9038** | **7.983** |

All rows use the same 1,500 photos. Re-evaluation reproduced every previous-model PSNR on the original 824 scenes exactly.

| Test group | Photos | Previous PSNR | New PSNR | Improvement, 95% CI |
|---|---:|---:|---:|---:|
| All | 1,500 | 24.127 | 24.446 | +0.319 [0.233, 0.400] dB |
| Original | 824 | 23.981 | 24.320 | +0.340 [0.234, 0.445] dB |
| Added | 676 | 24.305 | 24.598 | +0.293 [0.161, 0.417] dB |

The new model reduces newly clipped highlights by 0.063 percentage points (95% CI −0.089 to −0.040), while newly clipped shadows increase by 0.853 points (0.725 to 0.995).

The larger dataset also increases optimizer updates at the same epoch count. These are single-seed runs; intervals resample test scenes. The added group is reported separately because the original holdout had already been inspected. Comparison with published papers requires matching their split and preprocessing.

## Runtime and checkpoint

Training-function time was **4.70 minutes on one H100 80GB**, including cache loading and three evaluations. Estimated function compute was **$0.335**, excluding CPU preparation, image building, startup/idle time, storage, and account adjustments.

The final checkpoint has SHA256 `90781d57ff40e0cb1416bbde97b0179f9409340743f44f4da0e5dc8a6d00693c`. Native 720 × 480 flower and Paris exports passed the local check, including exact zero-strength bypass and source preservation. Installation instructions are in the [model card](../model-card.md).

[Training protocol](../full-fivek-training.md) · [Split registry](../../configs/splits/fivek_5000_7030.json) · [Metrics](full_fivek_summary.json)
