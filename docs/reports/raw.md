# RAW pilot results

The shared RAW model reaches **23.11 dB PSNR / 0.9075 SSIM** on 32 validation scenes. The pilot tests a scene-linear decoder and a 678,744-parameter editor with global tone controls and local exposure/contrast maps.

## Data and training

Original FiveK DNGs are decoded into float scene-linear sRGB. Expert C's 16-bit ProPhoto TIFFs are converted directly to float display sRGB through LittleCMS. Exact DNG crop and orientation metadata align the pairs.

Of 128 planned training scenes, **123** pass preparation; three fail RAW metadata checks, one has a fractional crop, and one has inconsistent geometry. All 32 validation scenes pass. Every scene comes from the earlier JPEG experiment's training pool. Validation selects the RAW variant; the JPEG benchmark uses different photos, targets, and preprocessing.

Four variants train from scratch for 800 updates each, batch 16, AdamW 0.0002 with cosine decay, and seed 20260919. Real-image order is matched. CNN training uses BF16 on H100; rendering and evaluation use FP32.

## Expert C validation

| System | PSNR ↑ | SSIM ↑ | RGB MAE ↓ | CIE76 ↓ |
|---|---:|---:|---:|---:|
| Neutral development | 16.32 | 0.7938 | 0.1395 | 19.99 |
| Global controls | 21.98 | 0.9006 | 0.0737 | 13.08 |
| Shared global + local | **23.11** | **0.9075** | **0.0646** | **12.14** |
| Shared + preservation training | 22.91 | 0.9039 | 0.0677 | 12.21 |
| Preservation + veil head | 22.53 | 0.9022 | 0.0698 | 12.36 |

Metrics average valid image content at 256 pixels. Neutral development uses the fixed shoulder and gamut mapping defined in the [model design](../raw-model-research.md).

The shared model reduces target MAE by **53.7%** from neutral development and gains **1.13 dB** over the learned global arm. A 5,000-resample paired scene bootstrap gives a 95% interval of **[0.26, 2.03] dB**. The SSIM difference interval crosses zero. All variants use one training seed.

## Local perturbations

Training faults use feathered ellipses with linear exposure/veil changes on every fourth batch. Evaluation uses a held-out diagonal-stripe support. Outside drift compares grades of the corrupted and clean versions of each image.

| System | Faulty-region target MAE ↓ | Outside drift ↓ |
|---|---:|---:|
| Global | 0.09365 | 0.00695 |
| Shared | 0.08761 | 0.01106 |
| Shared + preservation | 0.08539 | 0.01061 |
| Preservation + veil head | 0.09651 | 0.01173 |

Preservation training reduces shared-model outside drift by 4.1%; the interval for the change is [−0.00195, 0.00103]. The veil head worsens both target fidelity and robustness in this pilot. The shared model is the selected RAW variant.

## Export and native checks

A DNG-to-TIFF check on `fivek:a2339` produced a 2464 × 1632 uint16 sRGB TIFF while preserving the source hash. Scene-linear values ranged from −0.357 to 3.794 before display mapping.

A Mac MPS check on Canon ISO800 scene `fivek:a0007` used 12.17 MP output:

| Operation | Measurement |
|---|---:|
| Decode | 0.586 s |
| Crop and analysis | 0.240 s |
| Cached preview, median | 25.4 ms |
| Native 16-bit TIFF export | 0.958 s |

A second export took 1.539 s. Three native crops showed increased existing grain and color fringes; broad halos, banding, and tile seams were absent in those crops. Calibrated denoising and chromatic-aberration correction remain useful additions. Fujifilm RAF quality still needs a representative camera file.

## Runtime and follow-up

The four-arm H100 function took **108.1 seconds**, including cache loading, validation, and saves, with about **1.10 GiB** peak allocated GPU memory. Estimated function compute was **$0.13**, excluding preparation, interrupted attempts, cold starts, and verification.

The verification suite covered CPU/MPS parity, curve monotonicity, RAW headroom, crop geometry, cached inference, tiled rendering, and 16-bit TIFF output. The full suite at that stage passed 254 tests with one CUDA-only test skipped locally; the cloud training run exercised CUDA.

Next steps are representative Fuji checks, more RAW pairs, repeat seeds, and separate augmentation-only/consistency-only comparisons. See [RAW direction](../raw-direction.md), [decoder](../raw-pipeline-research.md), and [data preparation](../raw-data-research.md).
