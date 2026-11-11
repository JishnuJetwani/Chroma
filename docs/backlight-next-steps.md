# Backlight investigation

A strongly backlit JPEG exposed a LUT contouring problem in the original model. The app's optional **Backlight recovery** treatment smooths the color table, reduces local strength, and preserves bright source regions. Standard grading remains the default after the broader development comparison below.

## Diagnosis

The 640 × 360 source has **35.66% pure-white pixels**. The shirt and foreground retain texture, while the white sky contains little tonal variation.

The scene's learned LUT has negative RGB increments on **12.08%** of a 4,097-point neutral-ramp sweep. Off-diagonal table values produce oscillations under trilinear interpolation, causing gray contour islands near white. Local contrast approaches its 2× bound and makes the artifact more visible.

The recorded weighted monotonicity loss is only about 0.0024% of reconstruction loss. That scalar ratio motivates checking gradient norms and stronger shape constraints. The [architecture audit](research_backlight_architecture.md) and [training notes](research_backlight_training.md) cover the mechanism.

## Experimental treatment

The treatment reuses one cached recipe:

- Smooth the LUT residual with a five-cell Gaussian, sigma 1.0.
- Scale exposure and log-contrast maps by 0.5.
- Use 0.4 times the source's fifth-percentile minimum linear RGB as a black-floor correction, capped at 0.06, with a 0.025 soft toe.
- Blend toward the source across near-white values 0.78–0.98.

The supplied photo has smoother sky transitions and stronger foreground separation on visual inspection. Original-white interior drift falls from 0.02730 to zero, and foreground relative RMS contrast rises from 0.347 to 0.387. The pink optical ghost remains.

## Development comparison

The comparison uses **36 training-set scenes** at native 480p geometry: 12 bright-source cases and eight samples from each luminance tercile. Selection depends on source pixels. These checks measure regression on development examples.

| Variant | PSNR ↑ | SSIM ↑ | CIE76 ↓ |
|---|---:|---:|---:|
| Original model | 24.009 | 0.86990 | 8.509 |
| Tetrahedral interpolation | 23.978 | 0.86805 | 8.502 |
| Smoothed LUT | 23.824 | 0.87182 | 8.830 |
| Complete backlight treatment | 22.674 | 0.86732 | 9.535 |
| Shape-constrained LUT postfit | 23.917 | 0.87392 | 8.771 |

The complete treatment loses **1.335 dB** on average, especially on dark sources. It preserves source near-white pixels in all 23 eligible cases. These results support keeping it optional for severe backlight.

The LUT postfit uses [probe_lut_constraints.py](../scripts/probe_lut_constraints.py): 200 Adam updates at 0.0002 on copied tables, with fidelity and shape losses. It takes 10.90 seconds on CPU. Basis-table neutral reversals fall from 14.66% to 2.79%; the supplied photo's mixture falls from 12.08% to 1.38%. Near-white color drift increases, so further training needs neutrality anchors as well as shape constraints.

## Follow-up experiments

| Direction | Next comparison |
|---|---|
| [Tone](research_backlight_tone.md) | Broad base contrast, restrained veil subtraction, and highlight protection |
| [Architecture](research_backlight_architecture.md) | Absolute image statistics and constrained global tone curves |
| [Training](research_backlight_training.md) | Normalized shape losses, additive-light faults, and full-pipeline unchanged examples |
| [Semantics](research_backlight_semantics.md) | Photometric cues versus automatic and manual person/sky/clothing masks |
| [Hybrid inference](research_backlight_hybrid.md) | A small candidate set with source-based clipping and boundary checks |

The shape-loss and additive-veil helpers are implemented; the proposed JPEG retraining comparisons remain pending. The later [RAW pilot](reports/raw.md) tests monotone tone controls and scene-linear inputs.

Verification covered CPU/MPS parity, tiled finishing, cached treatment changes, native export, and source preservation. The original 158-test suite passed with one CUDA-only skip; 21 renderer/backlight checks passed after moving finishing into tiles.
