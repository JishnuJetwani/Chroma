# Backlight architecture audit

The original JPEG checkpoint produces gray contours in a bright sky through its learned LUT. The global-only output reproduces the artifact, making the color transform the first component to address.

## LUT response

On a 4,097-point neutral ramp, **36.7% of RGB slopes above 0.9 are negative**; full-range slopes span −2.92 to +6.61. Between input codes 200 and 255, a one-code increase can produce a 0.00902 output decrease or a 0.01937 increase.

Near-white diagonal knots are increasing. Trilinear interpolation also uses off-diagonal vertices, whose incompatible values create oscillations between knots. The LUT's white endpoint is about 0.980.

Tetrahedral interpolation removes the negative bright-neutral slopes for this table, with slopes 0.517–1.108. A separate 0.9–1.0 sweep measures 36.54% reversals for trilinear and zero for tetrahedral or Gaussian residual smoothing. Smoothing also moves the white endpoint: the five-cell filter changes it to about 0.952, motivating separate highlight protection.

The [ACES Common LUT Format](https://docs.acescentral.com/clf/specification/) describes tetrahedral partitioning along the main diagonal. Implemented helpers are in [lut_safety.py](../src/photograder/lut_safety.py), with tests for identity, affine transforms, grid vertices, gradients, smoothing, and CPU/MPS parity.

## Local controls and conditioning

The current local operator multiplies linear RGB according to exposure and log-luminance detail contrast. Broad additive color veils require a different operator. On the supplied photo, exposure ranges from −0.989 to +0.064 stops; contrast approaches its upper bound.

The decoder uses encoder features, style, and graded luminance. Adding pooled context or absolute image statistics is a compact way to test whether it needs better exposure cues. The existing `global_context` option adds masked means/stds to the LUT mixer with 64 extra weights.

The trained LUT bank is also highly correlated: one principal direction accounts for 97.64% of centered basis variance. The diverse initializer is a useful comparison before increasing the number of tables. [Context experiment](global-context-experiment.md).

## Preprocessing checks

Training uses LANCZOS resizing; the app's model preview uses bilinear resizing. On this photo, the difference changes the final rendering by only 0.0000845 RGB MAE. LUT behavior is the larger effect. Aspect-ratio padding covers 43.75% of this 16:9 analysis canvas and participates in encoder pooling/normalization; masking those statistics is another controlled comparison.

## Cost

Convolution/linear MACs for one 256-square analysis, with three styles:

| Architecture | Parameters | MACs |
|---|---:|---:|
| Global | 1,447,280 | 0.694G |
| Local | 1,711,058 | 1.205G |
| Preview RGB residual | 847,443 | 1.205G |
| Bilateral | 597,648 | 0.697G |
| Cascade | 3,142,100 | 3.103G |

These counts exclude interpolation, normalization, activations, LUT mixing, and rendering. Reducing LUT size from 33³ to 17³ removes 744,576 parameters. Increasing the CNN preview from 256 to 384 multiplies convolution cost by about 2.25.

## Experiments

1. Compare shape losses and monotone luminance curves using fixed data and updates.
2. Vary absolute context and LUT initialization independently.
3. Add one low-frequency affine/veil field if multiplicative controls remain insufficient.
4. Compare a two-LUT spatial mixture or a bilateral model with pooled context and learned guidance.
5. Separate predictor resolution from rendering/loss resolution and inspect native boundaries.

[HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/) and [spatial-aware LUTs](https://arxiv.org/abs/2108.08697) provide references for the spatial alternatives. Evaluate neutral ramps, saturated colors, white clothing, hair/sky edges, noise, and ordinary photographs alongside reconstruction scores. Completed inference experiments are in the [backlight report](backlight-next-steps.md).
