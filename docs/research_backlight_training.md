# Backlight training experiments

Standalone helpers in [training_constraints.py](../src/photograder/training_constraints.py) implement LUT shape/ramp losses and additive veil corruption. Trainer integration and the proposed JPEG comparisons are pending. The [backlight report](backlight-next-steps.md) records the completed inference and LUT-postfit experiments.

## Motivation

The original run trains global → repair → joint stages with RGB L1, LUT/map regularization, gamut loss, and preservation controls. Its synthetic faults cover exposure and detail contrast. Unchanged repair examples bypass the global LUT.

In one joint-training log entry, reconstruction is 0.05852548 and the weighted monotonicity term is 1.429e-6, about 0.00244% as large. The weighted curvature term is 7.580e-7. Log per-term LUT gradient norms to check how strongly these constraints influence training.

## Shape losses

With grid spacing `h=1/(D-1)`, test:

```text
Lmono  = mean(relu(-diff(LUT)/h))
Lcurve = mean((diff(diff(LUT))/h**2)**2)
```

This normalizes derivatives across LUT sizes. An unsquared hinge gives small reversals a stronger penalty. Evaluate dense neutral and near-neutral ramps, including luminance slope, RGB reversals, curvature, and highlight neutrality.

The [original adaptive-LUT implementation](https://github.com/HuiZeng/Image-Adaptive-3DLUT/blob/master/models.py) also uses shape regularization, with different scaling and loss conventions. A positive-increment luminance curve is a separate architecture comparison that enforces ordering directly.

## Additive-light data

Construct paired faults in linearized RGB by adding a nonnegative smooth light field. Retain the clean target, exact support, and newly clipped-channel availability.

| Family | Initial range | Held-out variation |
|---|---|---|
| Broad veil | Peak linear intensity 0.01–0.15 | Two off-frame sources |
| Off-frame bloom | Radial or elliptical falloff | Long anisotropic falloff |
| Local veil | Feathered finite support | Different smooth support geometry |

Start with neutral and mildly warm fields. Compare an unchanged/exposure-contrast/veil mix of 25%/25%/50% against the existing schedule at equal updates. Score recoverable pixels with reconstruction loss; evaluate clipped regions separately. Fix validation seeds by scene and variant.

[Wu et al.](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.html) motivates additive flare synthesis. [Flare7K](https://github.com/ykdai/Flare7K) offers additional layers for later work, primarily in nighttime settings.

## Unchanged examples and local restraint

Include same-style target-to-target examples through the full JPEG pipeline. Monitor source-to-target performance to avoid excessive identity behavior. On synthetic faults, supervise the clean and corrupted branches separately, with availability masks.

Compare outside-support reconstruction against the known target with consistency against the clean-input grade. Add contrast-magnitude control separately from spatial smoothness: a constant excessive map can have low total variation. Bright-neutral target losses can help anchor shirt and sky colors.

## Experiment order

1. Baseline versus normalized shape losses, with a fresh scene-grouped development split.
2. Existing faults versus added veil faults, keeping the chosen renderer fixed.
3. Multiplicative controls versus one bounded affine/veil field.
4. Existing conditioning versus absolute image statistics.
5. Joint training with and without full-pipeline unchanged examples.

Keep initialization, data order, effective batch size, and update counts matched within each comparison. Use fresh initialization when reserving validation scenes from a pool the old checkpoint already saw. Inspect ordinary exposures alongside severe backlight.

The helper tests cover derivative scaling at 17³/33³, identity, known reversals, gradients, exact support, and clipping availability. [The RAW pilot](reports/raw.md) subsequently tests a monotone curve and linear-veil training in a separate pipeline.
