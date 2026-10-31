# RAW grading model: research, implemented design, and evidence

2026-10-30. This is a separate scene-linear pipeline. The existing JPEG model,
renderer, and checkpoints retain their existing meaning.

## Why change the representation

The prior JPEG experiment exposed tonal reversals in a weakly constrained 3D
LUT, aggressive local contrast, missing absolute exposure cues, and no
additive-light training family. These measured findings are documented in
[the backlight investigation](backlight-next-steps.md). Switching file formats
alone cannot fix those failures; the new model changes the operator and input
domain as well.

Three primary research precedents guide the implementation:

- **HDRNet** predicts image-dependent transformation coefficients from a small
  preview and applies them to the original image. This supports keeping the
  expensive network independent of export resolution. Our renderer uses simple
  bilinear spatial fields rather than HDRNet's learned bilateral slicing.
  [Authors' project and paper](https://groups.csail.mit.edu/graphics/hdrnet/).
- **Exposure** develops RAW-derived linear RGB with understandable,
  differentiable filters. Its section 4 describes normalized prefix sums for
  monotone piecewise-linear curves, with parameters estimated at low
  resolution. Our model uses this established curve idea for luminance in a
  bounded shoulder coordinate; it does not reproduce their reinforcement or
  adversarial training procedure.
  [Primary paper, section 4](https://people.csail.mit.edu/hehaodele/projects/TOG18-exposure/paper.pdf).
- **How to Train Neural Networks for Flare Removal** models flare formation and
  builds synthetic clean/corrupted pairs. This supports explicit additive-light
  augmentation, with special treatment of saturation. A few smooth synthetic
  veils do not reproduce all lens ghosts, and their removal must be separately
  evaluated on real photographs.
  [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.pdf).

This architecture is an engineering combination of these ideas. It is not a
claim of a new research method or established improvement over the old model.

## Input contract and baseline development

`RawGradingModel` receives `B×3×H×W` FP32 **scene-linear sRGB**, after sensor
black/white normalization, demosaicing, camera white balance, and camera-to-sRGB
color conversion in the decoder. These fixed decoding stages are not learned.
The model has no denoising or detail-reconstruction network. Decoder identity,
settings, color transform, normalization, orientation, and clipping evidence
must be recorded with each dataset and checkpoint.

Values greater than one survive preview preparation and all scene-linear
adjustments. Negative values introduced by the camera color matrix are retained
until display gamut mapping. Neither one nor the maximum image sample is
assumed to be the sensor saturation threshold after white balance/color mixing.

The fixed baseline converts luminance `Y` to `max(Y,0)/(1+max(Y,0))`, scales RGB
by the luminance ratio, compresses only out-of-gamut chroma toward neutral gray,
and applies the sRGB encoding function. A neutral scene value of one becomes
approximately 0.735 sRGB, two is brighter, and four is brighter again.

**Strength zero means this neutral development.** A RAW array is not a display
image, and zero cannot mean showing its sensor codes directly. This baseline
is explicit and deterministic, but is not a reproduction of the camera JPEG
or an Adobe profile. A user exposure bias is applied in scene-linear stops
before the shoulder, including at zero model strength.

## Implemented model

[raw_model.py](../src/photograder/raw_model.py) contains the model, recipe,
renderer, loss, and augmentation helper. The default model has **678,744
parameters**, one Expert C style, a 256-pixel analysis preview, and spatial maps
whose longest side is 64 pixels.

The shared CNN consumes fixed-scale signed-log RGB, preserving negative colors
and headroom. Its global head also receives twelve absolute photometric
statistics: RGB log means and standard deviations, log-luminance mean and
standard deviation, per-channel fractions above one, and the fraction of
pixels containing a negative channel. Statistics exclude letterbox padding and
respect an optional validity mask. Group normalization cannot erase this
separate absolute-statistics path.

| Predicted control | Constraint and meaning |
| --- | --- |
| Global exposure | At most ±4 scene-linear stops |
| White balance correction | Zero-mean log gains, individually bounded within ±1 stop; correction on top of the decoder's camera white balance |
| Color matrix | `I + 0.15 P A P`, where `A=tanh(raw)` and `P=I−1lᵀ`; preserves both Rec.709 luminance and the gray axis |
| Saturation | Positive factor between 0.5 and 1.5, around linear luminance |
| Global luminance curve | 33 knots from normalized cumulative positive increments; endpoints zero and one |
| Local exposure | At most ±1.5 stops in a smooth low-resolution field |
| Local detail contrast | Log2 contrast multiplier within ±0.5, around a cached smooth source log-luminance base; detail residual limited to ±4 stops |
| Optional veil subtraction | Nonnegative RGB field at most 0.15 scene-linear units; disabled by default for the clean paired pilot |

For curve logits `z`, increments are
`softplus(clamp(z,−20,20)+log(exp(1)−1))+0.001`. Their normalized prefix sum
produces increasing knots. The input coordinate is `q=Y/(1+Y)`, not
`clamp(Y,0,1)`. This composition is monotone for a fixed global recipe and keeps
highlight ordering above scene-linear one. Strength in `[0,1]` interpolates
between the identity curve in this coordinate and the learned curve, retaining
monotonicity. Zero heads initialize exactly to baseline development up to
floating-point roundoff.

This guarantee concerns the **global luminance curve**. Different local
adjustments at different pixels, edits to arbitrary cached tensors, and
image-dependent reanalysis do not inherit a global ordering guarantee. Monotone
knots also do not guarantee attractive tone, skin colors, or absence of local
halos; validation must inspect these separately.

## Cached rendering API

```python
config = RawModelConfig(preview_size=256, map_size=64, styles=("Expert C",))
model = RawGradingModel(config)
recipe = model.analyze(linear_bchw, valid_mask=mask)
preview = render_raw(linear_bchw, recipe)
export = render_raw(native_linear_bchw, recipe, strength=0.8, exposure_bias_ev=0.5, tile_size=512)
```

The recipe supports `to`, `detach`, `state_dict`, and `from_state_dict`. Its
state dictionary is compatible with `torch.load(..., weights_only=True)`.
Maps retain the full source image coordinate system and aspect ratio. Output
tiles sample those maps using global coordinates; the renderer's remaining
operations are pointwise, so there are no tile-boundary filtering seams.
`return_aux=True` returns `(display_srgb, dict)` with `linear_display`,
`tone_luminance`, `pre_gamut`, and a per-channel `gamut_penalty`. At exactly zero
strength, the auxiliary dictionary is empty.

All model parameters and recipe/rendering arithmetic stay FP32. Optional CNN
BF16 autocast is limited to CPU/CUDA convolutional blocks. MPS uses FP32. The
implementation uses ordinary PyTorch convolutions, pooling, interpolation,
index selection, reductions, and matrix operations, with no custom CUDA kernel.

## Supervision and first experiment

`raw_supervised_loss(prediction, target, recipe, ...)` returns a dictionary with
a differentiable `total` and individually logged terms. Targets must be aligned
**display-sRGB** expert renditions. The starting objective is RGB L1 with small
target-gradient, map smoothness/magnitude, curve-slope-change, and bright-neutral
target terms. These weights are starting hypotheses rather than tuned results.
All non-L1 weights are explicit keyword parameters for a matched L1-only
control. Curvature is a prior; monotonicity does not depend on its weight.

`valid_mask` excludes invalid geometry. `available_mask` excludes unavailable
recovery supervision, including its gradient edges. In sensor data, a clipped
sensor channel can affect every display channel through demosaicing and the
camera color matrix. Propagate availability conservatively in the data pipeline;
a channelwise post-CCM threshold at one is not a valid substitute.

`additive_linear_veil` adds a supplied nonnegative light field to actual linear
RGB and returns clean target, support, clipping, and availability. By default
it applies **no clipping at one**. An explicitly supplied clipping level is a
controlled synthetic RGB experiment; newly clipped pixels are excluded in all
output channels. Field sampling and random seeds belong to the trainer.

Optional no-op prediction/target arguments are supported. They must come from
known scene-linear-to-display pairs. Feeding an expert sRGB image back into the
RAW model as though it were sensor-linear data would create incorrect no-op
supervision. A clean-input corruption-consistency branch is another valid
experiment, but should not be described as exact identity supervision.

Start with one expert and a deterministic, scene-grouped 512-pair training
subset, with validation scenes kept separate. Use actual RAW decodes, not
inverse-gamma JPEG surrogates. Keep the previous outer test closed while tuning.
Check native-resolution tone ramps, highlights, bright neutral regions, local
halos, export parity, and baseline-relative metrics before expanding the data
or enabling veil subtraction. A single pilot is feasibility evidence; it is not
a matched architecture comparison or proof of generalization.

## Verification completed

The first verification run passed 13 tests including CPU and actual Apple MPS
forward/backward parity. The tests exercise headroom above one, adversarial
curve logits, negative/all-zero/extremely bright input, bounded output, useful
head gradients, color-matrix invariants, nontrivial full-versus-tiled rendering,
recipe serialization, unavailable-pixel loss masking, and additive-light
clipping semantics. An initial MPS failure exposed unsupported non-divisible
adaptive area pooling; fixed pooling plus bilinear resampling removed that
operation. No fallback to CPU is needed for the tested renderer/model path.

Training outcomes belong in the run report once measured. These implementation
checks do not establish photographic quality.

## Local workbench integration

The app discovers `runs/raw-pilot-20261030/shared/model.pt` and opens a RAW tab
first when a trained compatible checkpoint exists. It retains the JPEG/PNG tab
and its original model. An explicit checkpoint can be passed with
`photograder app --raw-checkpoint PATH`; `run_app.command` prefers the new RAW
checkpoint once it is available. Missing RAW weights are displayed as missing,
without presenting a random initialization as a trained model.

`RawGrader` keeps the decoded native float image on CPU, creates its analysis
preview directly from native linear samples with antialiasing, and caches the
recipe. Slider changes and exports neither decode nor run the CNN again. DNG
files use the exact training `ActiveArea + DefaultCropOrigin/Size` crop before
EXIF orientation; proprietary formats use LibRaw's visible oriented image. This
is a metadata crop, with no target-dependent border guessing at inference.

Export transfers one 512-pixel tile at a time to the selected device, retaining
whole-image map coordinates. It collects final quantized tiles into a CPU
buffer; it never allocates a native floating output on the accelerator. PNG and
JPEG are 8-bit, while TIFF is actual 16-bit RGB from floating render values,
with an embedded sRGB ICC profile. Original files and existing exports cannot
be overwritten. The 60-megapixel decoder guard still applies.

The expanded verification run passed **63 tests** across the RAW model,
CPU/MPS parity, inference cache, native tiled export, true 16-bit TIFF, DNG
metadata geometry, app construction, decoder, and legacy JPEG inference. A
fixed-seed near-flat highlight ramp exposed a 2.4e-7 display reversal from gray
roundoff. Equivalent difference-based luminance/chroma arithmetic and a stable
bright shoulder removed it; the regression now requires no negative neutral
ramp increments. This numerical fix does not relax the monotonicity test.

Bilinear antialiasing matches prepared training data. In the installed Torch
2.8 MPS backend, antialiasing supports forward evaluation but not gradients to
input pixels. Model-parameter training works because source pixels are fixed;
input-gradient applications need a supported backend. The tested model uses no
automatic device fallback.
