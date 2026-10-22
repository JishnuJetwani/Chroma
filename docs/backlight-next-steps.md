# Backlight investigation and next experiments

2026-10-21. Five independent research directions, local renderer experiments,
one fixed-budget LUT optimization, and 36-image development regression checks.

## Outcome available now

The app has a **Treatment → Backlight recovery (experimental)** option. It uses
the existing trained checkpoint with a smoother color table, less aggressive
local adjustments, a gentle black-floor correction, and source-highlight
protection. Standard grade remains the default. The supplied photo is loaded
in the app with the experimental treatment selected, and PNG export was tested.

Original, previous model, experimental treatment (generated locally: `../runs/backlight_20261021/before_after.jpg`)

Full-size new output (generated locally: `../runs/backlight_20261021/backlight_final.png`) ·
photo diagnostics (generated locally: `../runs/backlight_20261021/photo_diagnostics.json`) ·
development comparison (generated locally: `../runs/backlight_20261021/development_evaluation/summary.json`)

The new result has a smoother sky and stronger foreground separation by visual
inspection. It still contains the pink optical ghost and intentional sun glow.
There is no expert target or human preference result for this photo, so this is
a candidate improvement, not a measured aesthetic optimum.

## What was wrong

1. **The learned LUT reverses tonal ordering.** On this photo's dense 4,097-point
   neutral ramp, 12.08% of RGB-channel increments were negative over the full
   range; the bright region was worse. A one-code input increase could cause a
   substantially larger output decrease. Off-diagonal cube values interacting
   under trilinear interpolation produce gray contour islands between otherwise
   increasing diagonal knots. Both residual smoothing and tetrahedral
   interpolation removed reversals on this particular neutral ramp; neither
   guarantees monotonicity for every LUT or color.
2. **The local module is too aggressive around intense light.** Its contrast map
   approaches its 2× limit, while darkening originally white areas creates gray
   shapes rather than recovering detail. About 35.66% of the supplied JPEG is
   exactly RGB white. The shirt and foreground retain substantial information.
3. **The training signal misses this failure.** The recorded weighted LUT
   monotonicity term was around 0.0024% of the reconstruction term. This is a
   scalar-loss comparison, not a gradient-norm measurement. Synthetic faults
   are multiplicative exposure/contrast changes; additive veil is absent.
   Unchanged repair examples bypass the global LUT and therefore do not teach
   the complete grader to leave a satisfactory photo alone.

The architecture itself is small enough. This evidence favors better operator
constraints and supervision before a larger backbone or a stack of specialists.

## Five research directions consolidated

| Direction | Useful conclusion | Decision |
|---|---|---|
| [Photographic tone](research_backlight_tone.md) | Separate broad tonal contrast from fine detail; retain clipped light sources; additive light differs from exposure. | Test restrained tone controls now. Aggressive veil subtraction crushed foliage and was rejected. |
| [Semantic hints](research_backlight_semantics.md) | Clothing/person/sky masks can limit edits, but backlit segmentation itself can fail. | Later compare photometric-only, automatic masks, and manual masks. Do not brighten an entire white shirt. |
| [Architecture](research_backlight_architecture.md) | Absolute photometric cues are missing; LUT instability dominates the resize mismatch on this photo. | Fix LUT behavior first; then test histogram/context inputs and one additional local operator. |
| [Hybrid system](research_backlight_hybrid.md) | Treat the learned grade as a proposal with explicit preservation checks. Lower clipping alone is a misleading quality score. | Ship an opt-in photographic treatment. Do not automatically apply it to every image. |
| [Training](research_backlight_training.md) | Normalize shape constraints, add additive veils, and train full-pipeline no-op behavior. | Utilities and a shape-only optimization probe are implemented; a new supervised training comparison is the next experiment. |

Primary precedents include [adaptive learned LUTs](https://arxiv.org/abs/2009.14468),
[additive flare synthesis](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.pdf),
and [guided filtering/dehazing](https://people.csail.mit.edu/kaiming/cvpr09/index.html).
These are established methods; this implementation does not establish novelty.

## What was actually tested

The supplied photo was rendered with the current joint model, global-only and
local-only controls, reduced strengths, several LUT smoothing widths, several
veil strengths, tetrahedral interpolation, and the installed external DeepLPF
checkpoint. DeepLPF also made the sky/shirt gray and was not selected for this
photo. Its training budget is unmatched and its performance here is qualitative.

The experimental policy is image-dependent but uses no hand-drawn shirt or sky
mask and no fixed image coordinates:

- Smooth the predicted LUT residual around identity with a 5-cell Gaussian,
  sigma 1.0. This smooths the color table, not image detail.
- Multiply the predicted exposure and log-contrast maps by 0.5.
- Estimate a conservative black-floor hint from the source preview's 5th
  percentile of minimum linear RGB; correction is 0.4 times that value, capped
  at 0.06. A soft toe with width 0.025 avoids hard shadow clipping. This is an
  aesthetic tone control, not an estimated physical flare field.
- Smoothly blend toward the original when minimum source RGB approaches white,
  over encoded values 0.78–0.98. Originally clipped white stays unchanged.
- Cache the recipe. Export uses the same recipe, whole-image coordinates, and
  pointwise finishing inside output tiles. Zero/zero strength bypasses exactly.

For this photo, correction magnitude is 0.04679 in the documented linear
coordinate system. The original-white interior's mean RGB drift was 0.02730
with the previous model and 0 with this treatment. Foreground relative RMS
contrast rose from 0.347 to 0.387; this describes contrast, not aesthetic quality.
No new zero-valued channels were created in this photo by either variant.

### Development regression results

All **36 scenes came from the previous model's training set**: 12 source-bright
cases and eight deterministic samples from each source-luminance tercile.
Selection used source pixels only. Targets and predictions were not used to
choose the cases. Metrics use matching native 480p geometry. No outer-test
images were reopened. These results cannot be compared directly to the earlier
824-scene benchmark or used as new generalization claims.

| Variant | PSNR ↑ | SSIM ↑ | CIE76 ↓ |
|---|---:|---:|---:|
| Current model | 24.009 | 0.86990 | 8.509 |
| Tetrahedral interpolation only | 23.978 | 0.86805 | 8.502 |
| Smoothed LUT only | 23.824 | 0.87182 | 8.830 |
| Complete backlight treatment | 22.674 | 0.86732 | 9.535 |
| Shape-constrained LUT postfit | 23.917 | 0.87392 | 8.771 |

The complete treatment is **not a universal replacement**: mean PSNR fell by
1.335 dB, especially on dark sources. Source near-white pixels were retained
exactly in all 23 eligible cases, but this preservation policy can differ from
the expert's intended edits. The original model remains the default.

### Actual optimization probe

`scripts/probe_lut_constraints.py` optimized only copies of the eight LUTs for
200 fixed Adam updates, learning rate 0.0002. It used fidelity to the original
tables plus normalized monotonicity/curvature and near-neutral luminance-ramp
constraints. **No photographs, expert targets, or test data were used.** It took
10.90 seconds on CPU. This is a shape-only postfit, not new supervised training.

Average basis-table neutral RGB reversal frequency fell from 14.66% to 2.79%.
On the supplied photo's resulting mixture it fell from 12.08% to 1.38%.
The checkpoint improved development SSIM slightly but worsened PSNR/CIE76,
and near-white color drift increased. It is retained as an experimental
checkpoint and is not installed as the app's main model. Stronger shape loss
alone is insufficient; future training needs color/neutrality anchors as well.

## Next training system and controlled experiments

Use the current shared CNN as the initial backbone. The preferred next system is
**absolute image statistics + a constrained global tone/color transform + small
local tone/veil controls + source-information guards**. A second segmentation
network is optional; its labels are hints, not instructions about desired light.

1. **Loss comparison first.** Create a new scene-grouped development split only
   within the 1,924 previous training scenes (for example 1,732/192). Train both
   controls from scratch because the old checkpoint has already seen those 192.
   Leave the frozen 70/30 files and 824-scene outer test unchanged. Compare
   baseline and normalized shape losses for 600 matched global updates on the
   same 512-scene subset. Add explicit neutral/chroma preservation and report
   gradient norms; do not use the postfit checkpoint as an unseen-data baseline.
2. **Data comparison.** Freeze the winning global branch and train two local
   branches for 600 matched updates: existing faults versus a mix containing
   broad additive veil, off-frame bloom, and unchanged photos. The new
   `additive_veil_corruption` helper supplies support and channelwise clipping
   availability. Recovery loss must exclude newly clipped channels.
3. **Capacity comparison if needed.** If additive-veiled training still fails,
   compare the existing multiplicative renderer with one bounded, smooth
   additive/affine field under identical data and updates. Add broad base-contrast
   control separately. Compare a monotone positive-increment luminance curve
   plus small color residual against a free 3D LUT; do not change all at once.
4. **Hints only when useful.** Add luminance percentiles, clipping fraction, and
   RGB statistics; then compare a frozen clothing/person helper with a manual
   mask ceiling on a few development scenes. Reject seam/halo and white-cloth
   regressions. MediaPipe's official multiclass model is small but explicitly
   warns of backlit failures, so low-confidence masks must fade toward fallback.
5. **Joint/preservation comparison.** Use 1,000 matched joint updates, including
   full-pipeline same-style no-op examples. Select on the new development set,
   fixed synthetic families, and randomized visual preference—not this photo
   alone. Run the final protocol once after settings are frozen; report the
   earlier outer-test reuse transparently and seek a fresh external set.

The normalized loss and additive-data utilities are implemented and tested, but
these full supervised training runs have **not** been launched. Today's work ran
locally without a new Modal GPU job or new dataset/model downloads.

## Verification and reproducibility

- Full test suite: 158 passed, one CUDA-only test skipped on this Mac. CPU/MPS
  LUT forward/backward and backlight output parity were tested.
- After moving finishing into tiles, 21 renderer/backlight tests passed again.
- The actual app analyzed the supplied photo, switched treatments with correct
  status, and exported a 640×360 sRGB PNG. Its MPS output differs from the CPU
  reference by at most one 8-bit code. The source checksum is unchanged.
- The experiment scripts, policy metadata, checkpoint, code/checkpoint hashes,
  per-case metrics and comparison images are in `runs/backlight_20261021/`.
- Original weights, dataset splits, and Standard grading behavior remain intact.
