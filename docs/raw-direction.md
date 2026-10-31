# RAW direction and physical-pipeline audit

2026-10-30. This consolidates the decoder/model audit before the paired RAW
pilot. Training results are deliberately left to the run report.

## Current system to measure first

Use the existing shared 679k-parameter RAW CNN, absolute photometric statistics,
monotone global luminance curve, and restrained local exposure/contrast maps.
Keep the optional veil subtraction disabled for the clean paired pilot. This
changes the parts implicated by the earlier JPEG investigation without adding
a segmentation or restoration network before there is evidence it is needed.

The network predicts a small recipe from a preview and applies it at native
resolution. This follows the efficiency principle of
[HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/), but uses bilinear maps,
not its bilateral slicing. Positive cumulative curve increments follow an
established construction in
[Exposure, section 4](https://people.csail.mit.edu/hehaodele/projects/TOG18-exposure/paper.pdf).
These are engineering precedents, not claims that this implementation reproduces
those papers or improves on the previous model.

## Audit findings and resolved issues

The current decoder is `raw-camera-float-wb-matrix-v2`. LibRaw subtracts each
sensor site's black before integer demosaicing. One common scale,
`raw.white_level − min(adjusted_black[0:4])`, preserves channel response ratios;
as-shot WB and the normalized inverse camera matrix then run in float32.
Per-channel whites describe sensor saturation and do not independently rescale
RGB. This follows LibRaw's
[scale_colors implementation](https://github.com/LibRaw/LibRaw/blob/master/src/postprocessing/postprocessing_utils_dcrdefs.cpp).
It fixes v1's unnecessary rejection of Canon files with one-code green black
differences. Missing as-shot WB still rejects explicitly. Cache rendering is
versioned `fivek-libraw-common-scale-dngcrop-lcms16-float-target-v3`.

The RAW model retains negative RGB and values above one until its intentional
display gamut mapping. The global tone coordinate is `Y/(1+Y)` for positive
luminance, so scene values above one remain distinguishable. Logarithm guards,
bounded detail residuals, and the final display clamp are not early RGB clipping.
The global curve is monotone for a fixed recipe; different local edits and
image-dependent reanalysis do not guarantee global pixel-order preservation.

Exact DNG crop tags matter: a0001's origin is `(15,7)`, not the rounded or
centered `(16,8)`. Training and native DNG app inference now share
`crop_dng_raw`. Native preview analysis now uses antialiased resizing, matching
the cache's resizing convention. Both corrections avoid teaching or applying
a transform on subtly different image content.

Visual and numerical checks found no gross WB/CCM defect. In the exact-cropped
a0001 comparison, float WB after demosaic and headroom-normalized LibRaw WB
before demosaic differ by display MAE `0.0007383`; Expert C's stronger contrast
and saturation are a separate development choice. The v2 Canon a3996 probe
retains `[-0.147739,1.858784]` and agrees with a LibRaw common-scale reference
to display MAE `0.00090584`. That reference had zero uint16 ceiling samples.
See probe artifacts (generated locally: `../runs/raw_20261030/decode_probe/`) and
[the detailed pipeline audit](raw-pipeline-research.md).

## Limits that affect interpretation

- This is not a fully floating-point ISP: LibRaw black subtraction clips
  below-black noise, and integer interpolation can clip excursions. The fixed
  daylight camera matrix is a reproducible approximation, not a full
  dual-illuminant DNG profile or Adobe look. A representative RAF from the
  user's exact Fujifilm model/compression remains untested.
- Model controls called white balance operate as gains in linear sRGB. They
  are residual color-balance controls, not exact changes to sensor WB. Exact
  camera-WB edits require camera-space gains or the conjugated transform
  `M diag(g) inverse(M)` with the camera matrix retained.
- The simple gamut operation contracts chroma toward equal-luminance gray.
  It guarantees bounded display RGB, not perceptual hue constancy. Compare
  saturated highlights with a stronger output-transform reference such as
  [ACES](https://docs.acescentral.com/system-components/output-transforms/).
- Aggregate sensor clipping fraction cannot identify recoverable pixels.
  Recovery experiments need a conservative per-pixel availability mask,
  propagated across demosaic and color mixing. A post-WB value above one is
  not proof of sensor saturation.
- Expert renditions supervise appearance, not scene radiance, lens-flare
  removal, or noise-free detail. Good reconstruction metrics alone do not
  establish these capabilities.

## Next systems, in order

1. **Matched tone controls.** Measure the shared model against its deterministic
   neutral RAW development; compare the current loss with L1-only supervision
   on the same scenes and update budget. Inspect native highlight ramps,
   saturated colors, skin, halos, and fine detail alongside held-out metrics.
   Keep the previously used outer test closed while tuning.
2. **Additive-light training.** Compare clean-only training against clean plus
   controlled broad linear veils and off-frame bloom, then enable bounded veil
   subtraction as a separate capacity comparison. Exclude unavailable clipped
   supervision. [Wu et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.pdf)
   supports semi-synthetic flare data; a smooth positive sRGB veil does not
   reproduce every physical spectrum, ghost, or sensor saturation pattern.
3. **Semantic hints only if errors demand them.** Compare a frozen person/sky
   helper, no helper, and a small manually labeled ceiling experiment. Fade
   uncertain masks and test boundary halos rather than assigning a desired
   brightness from the class label. [SegFormer](https://arxiv.org/abs/2105.15203)
   is a practical multiscale segmentation precedent, not evidence that masks
   will improve this grading task.
4. **Multiscale correction and denoising separately.** If broad exposure and
   detail conflicts remain, compare coarse-to-fine color/detail branches under
   matched training, as motivated by
   [Afifi et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Afifi_Learning_Multi-Scale_Photo_Exposure_Correction_CVPR_2021_paper.html).
   For actual RAW denoising, obtain appropriate paired or calibrated noise
   supervision rather than treating Expert C as noise-free truth;
   [Learning to See in the Dark](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Learning_to_See_CVPR_2018_paper.html)
   demonstrates a different, short/long exposure RAW training setup.

No new model family or pretrained weights were downloaded for this audit.
