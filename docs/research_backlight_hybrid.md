# Hybrid backlight inference

This proposal uses the trained model's cached recipe to generate a small set of candidate edits. Source-image diagnostics constrain those candidates, and visual review selects the useful treatment. The [backlight report](backlight-next-steps.md) records the implemented option and development results.

## Diagnostics

Compute masks once from the original image so every candidate is scored on the same pixels:

| Signal | Use |
|---|---|
| Near-white, low-texture interiors | Preserve source glow and avoid gray plateaus |
| Bright surviving texture | Allow mild contrast while monitoring luminance/chroma drift |
| Neutral and near-neutral LUT ramps | Detect reversals, excessive slope, and tint |
| Weak detail near the noise floor | Limit contrast amplification |
| Global/local recipe disagreement | Include reduced-strength candidates |

The source's pure-white fraction falls from 35.66% to zero under the original grade because those pixels become gray. Track newly introduced clipping and highlight drift separately.

## Candidate set

Reuse one `Grader.analyze` result. Begin with nine combinations of global/local strength from `{0, 0.25, 0.5}`, including identity. Then add source-highlight protection, one smoothed/projected LUT, and mild foreground tone or veil variants.

A source guard can use:

```text
output = source + correction_weight(source) * (candidate - source)
```

Keep masks in whole-image coordinates for export. Inspect the transition around hair and clothing as well as the flat sky interior.

For a fixed sampled ramp, blending `T` with identity gives slope `1-s+s*T'`. If the smallest slope is `m<0`, choosing `s <= 1/(1-m)` removes those sampled reversals. A separate upper-slope bound limits contrast gain. Full-color and local-map checks remain part of candidate review.

An alternative copies and projects the mixed LUT onto ordering constraints. Measure the projection's edit distance and color changes, and keep the checkpoint unchanged. [Exposure](https://people.csail.mit.edu/hehaodele/projects/TOG18-exposure/paper.pdf) provides context for a constrained action space and an explicit stopping option.

## Selection

Reject candidates with new contours, highlight tint, visible halos, crushed shirt folds, or substantial new clipping. Among the remaining edits, compare foreground detail and preservation of the original lighting. Prefer the smaller edit when differences are marginal.

No-reference image-quality scores can be a secondary diagnostic. Direct optimization can exploit them; [Liu et al.](https://arxiv.org/abs/2403.11397) documents that failure mode. A useful evaluation set includes snow, white clothing, silhouettes, fog, sunsets, and ordinary exposures.

## External comparisons

[DeepLPF](https://github.com/sjmoran/deeplpf-image-enhancement) supplies an alternative local-filter model. [IAT](https://github.com/cuiziteng/Illumination-Adaptive-Transformer) is another compact exposure-enhancement candidate. Pin source and weight revisions, normalize inputs through the same color/orientation path, and measure CPU/MPS runtime locally. IAT's demo uses CUDA calls that need adapting for Mac execution.
