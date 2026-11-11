# Selective local photo grading

*Pilot experiment draft. The later full-data benchmark is reported [separately](reports/full-fivek.md).*

## Abstract

We study a compact photo grader that predicts an image-dependent color LUT and local exposure/contrast maps. A shared CNN analyzes a preview, and a differentiable renderer applies the resulting controls at native resolution. On a 128-training / 32-validation / 32-test FiveK pilot, the local model improves expert-target agreement over a global baseline. An outside-consistency loss reduces drift under local input perturbations by 31.7%, with a 0.202 dB reconstruction cost. The experiment identifies a useful tradeoff between matching an expert rendition and stabilizing edits outside a perturbed region.

## Method

The single-style model has 1,711,026 parameters: four convolutional stages, eight learned 33³ LUTs, a 16-dimensional style embedding, and a shared decoder for 64-pixel exposure/contrast maps. Exposure is bounded to ±1 stop; contrast scales local log-luminance detail by 0.5–2.0. Guided upsampling aligns the maps to image boundaries.

Training first learns the global grade, then repairs known synthetic exposure/contrast faults, then optimizes both branches jointly. The preservation term compares outputs for clean and locally corrupted inputs outside a dilated corruption support. Both preservation variants share the same data and initialization.

The architecture draws on image-adaptive LUTs, local parametric filters, and preview-to-native rendering; references are collected in [related work](research.md#related-work).

## Experimental setup

The [frozen preview split](../configs/splits/fivek_preview.json) uses Expert C targets and keeps each scene's variants together. Training uses AdamW, learning rate 0.0002, weight decay 0.0001, FP32 MPS, batch two, accumulation four, and seed 42. Validation L1 selects checkpoints.

The global comparison allocates 400 optimizer updates. Each local curriculum allocates 160 global, 80 repair, and 160 joint updates. Selected checkpoint lineages contain 400, 352, and 368 updates for global, preservation, and preservation-disabled models respectively. CPU evaluation uses 256-pixel images, valid masks, PSNR, SSIM, CIE76, and held-out corruption geometry.

## Results

| Model | PSNR ↑ | SSIM ↑ | Outside drift ↓ |
|---|---:|---:|---:|
| Continued global | 18.9014 | 0.830520 | 0.000000000876 |
| Local, preservation | 19.9172 | 0.857706 | 0.00145317 |
| Local, preservation disabled | 20.1196 | 0.859866 | 0.00212794 |

Local with preservation improves PSNR over global by 1.016 dB (95% paired scene-bootstrap CI [0.434, 1.557]). Within the two local variants, preservation reduces outside drift by 0.000675 MAE [0.000281, 0.001102] and lowers PSNR by 0.202 dB [0.034, 0.360].

Two findings guide further work. The local system adds 0.427 percentage points of newly clipped highlights relative to global. Repair heads also continue to modify already edited targets, with MAE 0.0221 and 0.0168 for the two variants. Stronger unchanged-image supervision and LUT constraints are therefore useful next comparisons.

The [complete pilot results](results.md) include clipping, no-op tests, an external DeepLPF reference, style conditioning, native export timings, and confidence intervals. These measurements cover one seed and 32 test scenes. Larger comparisons and blind preferences remain future work.

## Reproduction

Follow the [data guide](data.md#rebuild-the-development-split) and run the [preview experiment matrix](experiments.md#replay-the-measured-preview-pilots). The [research overview](research.md) describes subsequent experiments; the [full FiveK report](reports/full-fivek.md) records the later 1,500-photo evaluation.
