# JPEG model card

Chroma predicts a global color LUT and local exposure/contrast maps from a photo preview. The renderer applies that recipe at the original resolution, with separate strength controls for the overall grade and local adjustments.

## Checkpoints

Both models have **1,711,058 parameters** and styles for FiveK editors C, A, and B. Expert C supplies most of the training data; A and B each have 17 training scenes.

| Checkpoint | Training scenes | Test scenes | PSNR | SSIM |
|---|---:|---:|---:|---:|
| Full FiveK | 3,500 | 1,500 | 24.45 dB | 0.9038 |
| Original JPEG, used in the flower and Paris examples | 1,924 | 824 | 23.98 dB | 0.8958 |

Each row uses its own Expert C holdout. The [full FiveK report](reports/full-fivek.md) compares both models on the same 1,500 photos.

SHA256 checksums:

- Full FiveK: `90781d57ff40e0cb1416bbde97b0179f9409340743f44f4da0e5dc8a6d00693c`
- Original JPEG: `1e60aaab4c95a9c3755a6d6f314c4c531b355d984dc8edab5c8f776c5e8d657a`

Weights are available on request at **jjetwani@uwaterloo.ca**. Save the supplied checkpoint as `checkpoints/jpeg.pt`, or pass its location with `--checkpoint`. The loader uses `weights_only=True` and checks the state dictionary against the model configuration.

## Training

The curriculum trains the global grade for ten epochs, local repair for five, and the joint model for ten. Both runs use AdamW, learning rate 0.0002, weight decay 0.0001, FP32, and batch eight on one H100. The final checkpoint follows the fixed schedule. All edits of a source photo share a partition.

Inputs and Expert C targets use the prepared 480p release from [Image-Adaptive 3D LUTs](https://github.com/HuiZeng/Image-Adaptive-3DLUT). Additional A/B targets come from official expert renditions. Evaluation uses valid image content on a padded 256-pixel canvas and a custom 70/30 split.

See the [full FiveK protocol](full-fivek-training.md), [original JPEG report](reports/jpeg.md), and [data guide](data.md). The full-data model has slightly more shadow clipping than its predecessor. Strong backlight can also expose LUT banding; the [backlight investigation](backlight-next-steps.md) documents that case.

The [packaging helper](../scripts/package_model.py) accepts the original JPEG checkpoint and checks its hash before creating a portable archive.

The code is unlicensed. Training data and derived materials retain the [FiveK research terms](https://data.csail.mit.edu/graphics/fivek/).
