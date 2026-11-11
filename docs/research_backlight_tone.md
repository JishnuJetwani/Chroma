# Backlight tone experiments

These proposed renderer comparisons target the surviving shirt and foreground detail in the backlit JPEG. The [completed investigation](backlight-next-steps.md) records which treatments were tested.

## Available detail

| Sample | Pure-white pixels | All channels ≥250 |
|---|---:|---:|
| Whole image | 35.66% | 45.57% |
| Top half | 55.95% | 69.51% |
| Shirt ROI `(316,163)` to `(446,344)` | 1.95% | 4.88% |
| Bottom quarter | 0% | 0% |

The sky has large constant-white regions, while the foreground and shirt retain tonal variation. Preserve the sky's glow and concentrate correction on recoverable contrast.

The [local renderer](../src/photograder/renderer.py) separates log luminance into guided base `b` and detail `d`, then applies `b + exposure + 2**contrast * d`. Broad tonal expansion and additive veil subtraction are separate controls worth testing.

## Highlight shoulder

For nonnegative luminance `z`, test knees `k = 0.65` and `0.80`:

```text
T(z) = z                                    when z <= k
T(z) = k + (1-k)*(z-k)/((1-k)+(z-k))         otherwise
```

Scale RGB by `T(Y)/Y`, then compress excess chroma toward equal-luminance gray. Compare with max-RGB normalization. Preserve source-white interiors through a feathered blend and inspect shirt-fold gradients, tint, and boundary halos.

[Reinhard et al.](https://www-old.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf) provides tone-mapping context; [ACES](https://docs.acescentral.com/system-components/output-transforms/) provides a more developed color/gamut reference.

## Broad base contrast

Following the base/detail separation in [Durand and Dorsey](https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/), test:

```text
b = GuidedFilter(log2(Y))
d = log2(Y) - b
output = b + exposure + w*(base_gain-1)*(b-pivot) + (detail_gain-1)*d
```

Start with base gains 1.15/1.30, detail gain 1.10, pivot `log2(0.35)`, and guided radii 13/32 at width 640. Use a fixed feathered foreground support for the first comparison. Inspect noise, hair halos, and overly sharp fabric. Larger filters require matching tile overlap.

## Additive veil

Test `J=max(I-w*V,0)` in linearized RGB with neutral `V` values 0.01, 0.025, and 0.04. Compare against exposure-only edits matched in foreground mean luminance. A later model could predict a smooth 4×4 or 8×8 veil field.

[Flare-removal work](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.html) motivates additive synthesis. JPEG linearization remains an approximation to the original camera processing. Strong subtraction can crush foliage or tint clothing; a smooth field also leaves structured flare ghosts.

Use the fixed controls to identify helpful operators before training another head. Include clean, high-key, white-object, and intentionally soft photographs in the comparison.
