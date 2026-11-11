# Research and architecture

Chroma combines a learned global grade with local exposure and contrast adjustments. The research question is how well a compact model can repair a local exposure fault while preserving its chosen style elsewhere.

## System

A four-stage CNN with 32/64/128/128 channels analyzes a 256-pixel preview. A 16-dimensional style embedding conditions a mixture of eight 33³ RGB LUTs and a shared spatial decoder. The decoder predicts 64-pixel exposure and log-contrast maps, with skip connections and globally graded luminance as inputs.

The renderer applies exposure within ±1 stop and detail contrast between 0.5× and 2× around a guided log-luminance base. Guided upsampling aligns the maps with image edges. Gather-based LUT interpolation supports CPU, MPS, and CUDA. One cached recipe serves both interactive previews and tiled native-resolution export.

The three-style model has 1,711,058 parameters. Internal alternatives include a fixed LUT, a global-only network, cascaded specialists, a preview RGB-residual network, and a bilateral grid.

## Related work

| Work | Relevant idea |
|---|---|
| [HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/) | Preview analysis and full-resolution spatial transforms |
| [Image-Adaptive 3D LUTs](https://arxiv.org/abs/2009.14468) | Learned LUT bases with image-dependent mixing |
| [DeepLPF](https://arxiv.org/abs/2003.13985) | Editable local parametric filters |
| [Spatial-aware 3D LUTs](https://arxiv.org/abs/2108.08697) | Image-level and pixel-level LUT mixing |
| [MonetGPT](https://arxiv.org/abs/2505.06176) | Retouching supervision from synthetic edit puzzles |
| [Modular Neural ISP](https://arxiv.org/abs/2512.08564) | Modular rendering and style conditioning |

## Training and evidence

Training progresses through global grading, synthetic local repair, and joint optimization. Recoverable exposure/contrast faults have known spatial support; unchanged examples encourage restraint. An outside-consistency loss compares grades of a clean and corrupted input at the same style.

The [early controlled pilot](results.md) measured a 31.7% reduction in outside-region drift from preservation training, alongside a 0.202 dB PSNR cost. The [full FiveK model](reports/full-fivek.md) reaches 24.45 dB / 0.9038 SSIM on the custom 1,500-photo holdout. A separate [RAW pilot](reports/raw.md) tests scene-linear decoding and monotone tone curves.

## Next comparisons

1. Train on a public FiveK protocol with matched preprocessing and evaluation.
2. Repeat the preservation comparison across three seeds and a larger development set.
3. Test LUT shape constraints and full-pipeline unchanged-image supervision, following the [backlight findings](backlight-next-steps.md).
4. Compare guided/bilinear maps, shared/cascaded encoders, and tonal specialists under matched budgets.
5. Add aligned S24 local-tone targets once acquisition is practical.

Evaluation combines expert-target metrics, known-region recovery and outside drift, native-resolution artifact inspection, and blind preferences. A proposed preference study uses 100 held-out photos and three raters per photo. Human ratings remain to be collected.
