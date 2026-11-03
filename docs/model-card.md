# JPEG model card

## Intended use

Local JPEG/PNG photo grading for a personal research project. The model predicts a global color LUT and spatial exposure/contrast maps. It cannot restore image detail destroyed by clipping or guarantee a preferred aesthetic.

## Checkpoint

- Local source: `runs/full-7030-20261012/joint/model.pt`.
- Portable installation: `checkpoints/jpeg.pt`.
- SHA256: `1e60aaab4c95a9c3755a6d6f314c4c531b355d984dc8edab5c8f776c5e8d657a`.
- Parameters: 1,711,058.
- Styles: FiveK Expert C, A, B. Expert C is the recommended style; A/B have very little training data.
- Format: PyTorch tensor-only checkpoint, loaded with `weights_only=True` and strict state-dictionary checks.

No checkpoint is hosted yet. A fresh source checkout requires a separately supplied checkpoint or a training run. To package the existing weights for a future release:

```bash
uv run --no-sync python scripts/package_model.py \
  --checkpoint runs/full-7030-20261012/joint/model.pt \
  --output dist/selective-grade-jpeg-v0.1.0.zip
```

The archive contains `checkpoints/jpeg.pt`, its checksum/configuration, and this model card. Extract it into the project root. Packaging does not upload or publish the archive.

## Training and evaluation

The prepared FiveK collection contains 2,748 scenes. The custom frozen 70/30 split has 1,924 training and 824 test scenes; all editor variants stay with their source scene. There are 1,958 training pairs after including a small matched editor A/B subset.

The curriculum is ten global epochs, five synthetic local-repair epochs, and ten joint epochs. The final checkpoint is selected by the fixed schedule without test-based tuning. Training used a single H100, FP32 rendering, and batch eight. The recorded GPU function work was about 364 seconds; this excludes data preparation and is not a bill.

The primary Expert C test result is **23.9808 dB PSNR, 0.8958 SSIM, and 8.346 CIE76 color difference**. These are reference-matching metrics, not quality percentages. Evaluation uses padded 256-pixel inputs with valid-content masks. See [the report](reports/jpeg.md) and [machine-readable results](reports/jpeg-metrics.json).

## Limits and provenance

The global comparison has fewer training updates. The run does not establish causal benefits from preservation training, convergence, superiority over published methods, or broad camera generalization. Newly clipped highlight pixels increased relative to the global baseline. Brightening alone can explain much of the benefit on some examples.

Inputs and Expert C targets came from the prepared 480p FiveK release documented by [Image-Adaptive 3D LUTs](https://github.com/HuiZeng/Image-Adaptive-3DLUT); the additional A/B targets came from official expert renditions. The source files and their checksums remain in local dataset manifests. Split identities are retained in `configs/splits/cloud_full_7030.json`.

The code is currently unlicensed. Dataset terms and attribution requirements remain separate; no commercial-use permission for these weights is granted by this project. Consult the original [FiveK terms](https://data.csail.mit.edu/graphics/fivek/).
