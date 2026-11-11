# RAW model

The RAW model predicts editable controls from scene-linear RGB, preserving highlight headroom until display rendering. It has **678,744 parameters**, a 256-pixel analysis preview, and spatial maps with a 64-pixel longest side. The [pilot report](reports/raw.md) compares four training variants.

## Input and neutral development

`RawGradingModel` receives `B×3×H×W` float32 scene-linear sRGB after demosaicing, normalization, camera white balance, and color conversion. Matrix-induced negatives and values above one remain available to the renderer.

Neutral development maps positive luminance through `Y/(1+Y)`, scales RGB, contracts out-of-gamut chroma toward equal-luminance gray, and applies sRGB encoding. Model strength zero uses this baseline. Exposure bias is applied in scene-linear stops before the shoulder.

## Predicted controls

The CNN reads signed-log RGB. The global head also receives twelve absolute statistics: RGB log means/stds, log-luminance mean/std, per-channel fractions above one, and the fraction of pixels with a negative channel. Statistics exclude padding and invalid pixels.

| Control | Bounds |
|---|---|
| Global exposure | ±4 stops |
| Residual color balance | Zero-mean log gains, each within ±1 stop |
| Color matrix | `I + 0.15 P A P`, with `A=tanh(raw)` and `P=I−1lᵀ`; preserves luminance and gray |
| Saturation | 0.5–1.5 around linear luminance |
| Luminance curve | 33 increasing knots, endpoints zero and one |
| Local exposure | ±1.5 stops |
| Local detail contrast | Log2 multiplier ±0.5; detail residual bounded to ±4 stops |
| Optional veil | Nonnegative RGB field up to 0.15 scene-linear units |

Curve increments use `softplus(clamp(z,−20,20)+log(exp(1)−1))+0.001`. Normalized cumulative sums produce monotone knots in the coordinate `q=Y/(1+Y)`. Mixing this curve with identity preserves monotonicity for a fixed global recipe. Local fields can change pixel ordering independently.

The veil head is disabled in the selected shared checkpoint. Zero-initialized controls reproduce neutral development up to floating-point roundoff.

## Rendering API

Given a decoded scene-linear tensor `linear_bchw` and validity mask `mask`:

```python
from photograder.raw_model import RawGradingModel, RawModelConfig, render_raw

model = RawGradingModel(RawModelConfig(preview_size=256, map_size=64, styles=("Expert C",)))
recipe = model.analyze(linear_bchw, valid_mask=mask)
preview = render_raw(linear_bchw, recipe)
output = render_raw(native_linear_bchw, recipe, strength=0.8,
                    exposure_bias_ev=0.5, tile_size=512)
```

This example constructs an initialized model; trained inference uses `RawGrader` with a checkpoint through the [app](usage.md#raw-developer). Recipes support device transfer, detaching, and tensor-only serialization. Tiles sample maps in whole-image coordinates. `return_aux=True` exposes display-linear RGB, tone luminance, pre-gamut RGB, and gamut penalties.

Rendering and recipes use FP32. Optional BF16 covers CPU/CUDA CNN blocks; MPS uses FP32. Native export keeps the float source on CPU and processes 512-pixel tiles on the device. PNG/JPEG exports are 8-bit; TIFF exports are 16-bit RGB with sRGB ICC metadata.

## Supervision

`raw_supervised_loss` combines RGB L1 with target gradients, map smoothness/magnitude, curve curvature, and bright-neutral terms. Invalid geometry and unavailable recovery pixels are masked, including affected gradient edges.

`additive_linear_veil` adds a nonnegative light field and returns support and availability metadata. An optional synthetic clipping level excludes newly clipped pixels from recovery supervision. Clean/corrupted consistency compares the two rendered outputs. Sensor-space saturation masks require propagation through demosaicing and color mixing.

The shared model, preservation training, and veil head were compared on the same [123/32 split](reports/raw.md). The shared model scored best on expert-target accuracy.

## Implementation checks

Tests cover curve ordering under extreme logits, headroom, negatives, finite gradients, color-matrix invariants, CPU/MPS parity, serialization, loss masks, tiled output, and TIFF precision. Stable luminance/chroma arithmetic removed a 2.4e-7 neutral-ramp reversal found during verification.

MPS supports the antialiased preview forward path used here; input-pixel gradients through that operation require another backend. Model-parameter training uses fixed source pixels.

The design draws on [HDRNet's preview-based transforms](https://groups.csail.mit.edu/graphics/hdrnet/), [Exposure's monotone curves](https://people.csail.mit.edu/hehaodele/projects/TOG18-exposure/paper.pdf), and [additive flare synthesis](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.html).
