# Development results

The latest benchmark is the [full FiveK experiment](reports/full-fivek.md): **24.45 dB PSNR / 0.9038 SSIM** on 1,500 photos. This page records the earlier pilots that informed the model.

## 256-pixel pilot

The frozen [preview split](../configs/splits/fivek_preview.json) contains 128 training, 32 validation, and 32 test scenes. Training uses seed 42 and validation-L1 checkpoint selection. CPU evaluation uses encoded sRGB, valid-mask SSIM v2, CIE76, and the `local-recoverable-v2-headroom` corruption protocol.

| Model | Allocated updates | PSNR ↑ | SSIM ↑ | CIE76 ↓ | Outside drift ↓ |
|---|---:|---:|---:|---:|---:|
| Fixed LUT | 160 | 17.9742 | 0.806807 | 16.7206 | 0 |
| Global, initial | 160 | 18.1768 | 0.813748 | 16.2530 | 0.0000000203 |
| Global, continued | 400 | 18.9014 | 0.830520 | 14.5765 | 0.000000000876 |
| Local, preservation | 400 | 19.9172 | 0.857706 | 12.9404 | 0.00145317 |
| Local, preservation disabled | 400 | 20.1196 | 0.859866 | 12.7352 | 0.00212794 |

The continued global model adds 240 updates. Each local curriculum allocates 160 global + 80 repair + 160 joint updates. Validation selects lineages containing 400, 352, and 368 updates for continued global, local with preservation, and local without preservation respectively. Global has 1,447,248 parameters; local has 1,711,026.

| Candidate minus continued global | PSNR difference, 95% CI | SSIM difference, 95% CI | CIE76 difference, 95% CI |
|---|---:|---:|---:|
| Local, preservation | +1.016 dB [0.434, 1.557] | +0.02719 [0.01980, 0.03482] | −1.636 [−2.163, −1.084] |
| Local, preservation disabled | +1.218 dB [0.592, 1.819] | +0.02935 [0.02103, 0.03784] | −1.841 [−2.429, −1.213] |

Both local variants improve expert-target agreement, including within corrupted regions. Their change in the additional error caused by corruption has intervals crossing zero. Both also increase outside and boundary drift relative to the global model. The early global checkpoint uses nearly constant LUT mixing weights; [context and initialization](global-context-experiment.md) are follow-up experiments.

## Preservation and clipping

With architecture, data, curriculum, and initialization held fixed, preservation training reduces outside drift by **31.7%**, or **0.000675 MAE** (95% CI [0.000281, 0.001102]). This is about 0.172 of an 8-bit level. Clean-image PSNR falls by **0.202 dB** [0.034, 0.360]. Region-recovery and boundary-drift intervals cross zero.

Against continued global, local with preservation adds **0.427 percentage points** of newly clipped highlights [0.210, 0.687], 1.185 points of all-white pixels, and 0.002300 boundary MAE [0.001338, 0.003348]. Clipping is measured before output quantization at encoded-sRGB full scale, excluding already clipped input pixels.

These intervals use 2,000 paired scene-bootstrap samples from one training seed and 32 scenes. The comparison covers the complete curricula, whose parameter counts and compute differ.

## Already edited targets

The repair heads processed the same 32 Expert C targets with global grading bypassed. Local strength was one, global strength zero.

| Repair head | No-op MAE ↓ | Pixels changed by more than 2/255 in any channel |
|---|---:|---:|
| Preservation | 0.0220640 | 70.02% |
| Preservation disabled | 0.0168046 | 69.60% |

Disabling preservation reduces mean change by 0.0052594 MAE [0.0029131, 0.0074551]. The changed-pixel difference is −0.42 percentage points [−7.72, 8.24]. Both heads continue editing many pixels in these clean targets. This motivates stronger unchanged-image supervision alongside the outside-drift objective.

## External DeepLPF reference

An author-repository checkpoint was scored on the same 32 scenes and input tensors. Their IDs appear in its recovered DPE test list and are absent from the corresponding training/validation lists; those lists came from a third-party mirror.

| Metric | Mean | 95% scene-bootstrap CI |
|---|---:|---:|
| PSNR ↑ | 24.8813 dB | [23.4658, 26.2544] |
| SSIM ↑ | 0.931890 | [0.915849, 0.946234] |
| CIE76 ↓ | 8.24634 | [7.16772, 9.46814] |
| Outside drift ↓ | 0.0108331 | [0.0082711, 0.0139580] |

The external checkpoint uses a larger training budget: 2,250 images and validation selection at epoch 774, compared with 128 images here. It provides a quality reference under unmatched training. Its higher target agreement and higher outside drift show why both measures are useful.

## Three-editor pilot

The 1,711,058-parameter extension adds named A/B embeddings to the C model. It uses 16 training, four validation, and four test scenes, each with three editor targets. Training allocates 120 updates over 20 epochs; validation selects update 114.

| Editor | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| C | 0.07214 | 21.92 | 0.8871 |
| A | 0.07983 | 20.80 | 0.9014 |
| B | 0.05480 | 23.66 | 0.9407 |

Changing style produces average differences of 0.78–2.12 RGB levels out of 255, smaller than the human-target differences. More examples per editor are the next step for stronger style separation.

## Mac measurements

Hardware: M4 MacBook Air, 24 GB, Python 3.11.14, PyTorch 2.8.0/MPS.

| Check | Measurement |
|---|---|
| 200 training steps, batch two, 512px rendering | Mean 0.923 s; median 0.792 s |
| End-of-run Metal allocation for training | 1.35 GB |
| Cold analysis of a 12.05 MP image | 3.108 s |
| Cached 681 × 1024 preview | Median 0.247 s |
| Native PNG exports | 13.149 / 8.624 / 8.753 s |
| End-of-run Metal allocation for export | 1.24 GB |

Export times include tensor conversion, tiled rendering, and encoding. Initial decoding/model loading are excluded. Measurements use one fixture on a shared host; memory values are end-of-run allocations.

A separate 16-pair learning check reduced training loss from 0.1132 to 0.0493 and four-scene validation L1 from 0.1278 to 0.0817 over 20 passes.

For later runs, see the [824-photo JPEG benchmark](reports/jpeg.md), [full FiveK benchmark](reports/full-fivek.md), and [RAW pilot](reports/raw.md). Replay instructions are in the [experiment guide](experiments.md).
