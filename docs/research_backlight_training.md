# Backlight research: training, losses, and data

Research date: 2026-10-21. This memo proposes experiments; it does not report a newly trained model or a real-photo quality improvement. The supplied backlit JPEG is a development example with no reference edit. It must not become the validation set. Its washed foreground is consistent with veiling light, but the optical cause cannot be established from this one processed JPEG.

Implementation status: [training_constraints.py](../src/photograder/training_constraints.py) supplies opt-in `lut_shape_losses`, `lut_ramp_losses`, and `additive_veil_corruption`. They are not connected to the current trainer. [Sixteen tests](../tests/test_training_constraints.py) passed locally, covering derivative scaling at 17³/33³, identity, a known ramp reversal, gradients, exact support, and channelwise clipping availability. This validates the helpers, not the proposed learning outcome.

## Recommendation

Run the cheapest two interventions separately: repair the LUT shape constraints, then teach a small model about additive veils using existing training images. Retain supervised Expert C pairs and deliberately include clean images. A larger network cannot compensate for a renderer that lacks an appropriate operation, and a better average SSIM does not establish that white clothing, faces, and bright light sources are handled acceptably.

Do not immediately add several perceptual losses, a new backbone, and flare augmentation together. Their effects would be inseparable. Start from scratch for the new model comparison because the existing full-data checkpoint has already seen any validation photos carved out of its 1,924 training scenes.

## What the repository actually trains

The full run uses 10 global, 5 repair, and 10 joint epochs, at 256-pixel rendering, without validation-based selection. It has 1,924 training scenes and 824 outer-test scenes. The Expert C test SSIM of 0.8958 is a recorded historical result, not a new measurement in this investigation. See [the original report](reports/jpeg.md).

The relevant behavior is in [training.py](../src/photograder/training.py), [models.py](../src/photograder/models.py), [corruptions.py](../src/photograder/corruptions.py), and [renderer.py](../src/photograder/renderer.py):

| Existing choice | Consequence for this failure |
|---|---|
| RGB L1 is the reconstruction loss. | Provides a useful color/fidelity anchor, but small objectionable regions can contribute little to its mean. |
| The LUT bank uses squared negative first differences and squared second differences, averaged across all table cells. | Small, localized reversals receive weak penalties. The coefficient does not directly represent a derivative because grid spacing is omitted. |
| Joint training overrides recipe LUT penalties with bank penalties. | The bank has second-difference curvature, while `recipe_penalties` calls a first-difference energy “smoothness.” These are different objectives. |
| Global coefficients are softmax weights. | If every basis obeys a linear monotonicity constraint, their convex mixture also does. A penalty alone provides no such guarantee. |
| Local exposure is bounded to ±1 stop; local detail contrast is 0.5–2×. | These operations do not explicitly subtract additive colored veil. Amplifying detail can reveal JPEG artifacts and does not restore lost base contrast. |
| Repair corruptions are recoverable multiplicative exposure or local log-luminance contrast faults. | No additive light, lens-flare layer, or explicit clipped-input case is trained. Combinations and exact exposure inverse supervision are implemented but disabled in the full run. |
| Repair identity cases and the joint identity branch bypass the global LUT. | They teach local no-op behavior; they do not teach the complete grader to leave a satisfactory image alone. |
| Joint counterfactual preservation compares unaffected pixels to a detached current-model output. | It discourages new drift, but the reference may itself be an undesirable grade. |
| Map total variation penalizes spatial change. | A uniformly excessive contrast map can have zero total variation. This term does not discourage excessive adjustment magnitude. |

There is a measurable scale issue. At joint epoch 10, step 2300, the saved log reports reconstruction 0.05852548, LUT monotonicity 0.000142917, and LUT smoothness 0.000758031. Their configured weights are 1, 0.01, and 0.001. Thus the weighted monotonicity term is 1.429e-6, about 0.00244% of reconstruction; curvature contributes 7.580e-7. Scalar magnitudes alone do not prove gradient dominance, but this is a strong reason to log per-term gradient norms before increasing capacity. Source: saved log (generated locally: `../runs/full-7030-20261012/modal-20261013T004139Z.log`).

A read-only CPU inspection of the final checkpoint found a bank monotonicity penalty of 0.000154805 and curvature penalty of 0.000819299. Across all basis/channel/axis entries, 36.0% of first differences are negative. This fraction includes cross-channel color derivatives, so it is **not** a 36% visible-band rate; the separately observed bright neutral-ramp reversals are the more directly relevant diagnostic. No outer-test images were reopened for these checks.

## 1. Make LUT shape an explicit engineering objective

Let the LUT spacing be `h = 1 / (D - 1)`. Begin with:

```
Lmono  = mean_axes mean_cells relu(-diff(LUT) / h)
Lcurve = mean_axes mean_cells (diff(diff(LUT)) / h**2)**2
```

The non-squared hinge continues to penalize tiny reversals more strongly than a squared hinge. Normalization makes coefficients more interpretable across 17³ and 33³ tables; it does not guarantee matching discretization error for arbitrary nonsmooth tables. Curvature is zero for an affine transform, so it does not flatten ordinary linear contrast. First-difference energy instead penalizes slope itself. Monitor both raw and weighted losses, derivative quantiles, and gradient norms on the LUT parameters.

The original image-adaptive LUT method supplies direct precedent for shape regularization. Its official code uses first-difference smoothness and an **unsquared** negative-difference hinge. Its paired training script defaults to monotonicity coefficient 10 with MSE, different bank/mixer conventions, and no derivative normalization. Copying that coefficient into this repository would therefore not reproduce its objective. [Original model code](https://github.com/HuiZeng/Image-Adaptive-3DLUT/blob/master/models.py), [official paired trainer](https://github.com/HuiZeng/Image-Adaptive-3DLUT/blob/master/image_adaptive_lut_train_paired.py).

Add a small synthetic diagnostic independent of photographs: evaluate each basis and each development-image mixture on a dense full-range neutral ramp and six near-neutral RGB rays. Penalize negative luminance slopes; report individual RGB reversals and large second differences as well. Preserve floating output for this check, because clipping can hide reversals. Use a fixed ray grid for every experiment. Dense sampling detects bad regions but is not a mathematical guarantee between samples or away from those rays.

For a guarantee, a small positive-increment monotone 1D luminance curve is easier to constrain than a free 3D color transform. That is an architecture control, not a loss-only ablation. Full RGB coordinatewise monotonicity is stronger than necessary and can limit legitimate color mixing; prioritize luminance ordering on neutral/near-neutral rays and verify color performance before enforcing every cross derivative. A projected monotone LUT should also be considered only as an explicit alternative: projection changes model capacity and the optimum.

## 2. Add an actual additive veil family

Use the existing training targets as clean photographs and synthesize `x = encode(clip(decode(y) + F))`, where `F >= 0` is a smooth linear-light RGB veil. Preserve `y`, the sampled veil, support, and **channelwise newly clipped availability**. Keep image creation outside the model so paired ground truth stays exact wherever the corruption is recoverable. Approximate sRGB linearization is a controlled synthetic construction; it does not undo the unknown camera processing of a JPEG.

The physics-motivated flare work of Wu et al. composes clean images, flare, and noise additively in approximately linear space. It also treats saturated light-source regions specially instead of requiring the network to invent their hidden scene content. This supports the corruption family and masked supervision, not an assertion that a few procedural masks reproduce real lens optics. [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.pdf).

Cheap first families:

| Family | Proposed training distribution | Held-out development variation |
|---|---|---|
| Broad veil | Constant or low-frequency RGB field, peak linear intensity 0.01–0.15 | A mixture of two off-frame sources |
| Off-frame bloom | Smooth radial/elliptical falloff with source beyond an image edge | Long anisotropic falloff and opposite-edge location |
| Local veil | Feathered field with exact finite support | A different smooth support construction |

These ranges are starting hypotheses, not measured lens parameters. Start with mostly neutral and mildly warm veils; sample the source location independently of whether a person is present. Include white clothing, bright skies, snow, walls, low-key scenes, and genuinely acceptable existing highlights among the clean training negatives. Randomly creating flares only on portraits would teach an unwanted shortcut.

Keep a severity bin with no newly clipped channels for exact reconstruction. A second, stronger bin may include clipping, but exclude newly clipped channels from recovery error and evaluate them separately. Do not quietly multiply every veil by global headroom: a single white pixel can otherwise erase the intended corruption. A clean target retained in a dictionary is **not** permission to optimize its unrecoverable texture; downstream loss must use the availability mask. For the clipped area, a conservative observed-highlight retention term is a distinct objective, not texture-recovery supervision.

For no added data, proposed synthetic minibatch proportions are 25% unchanged targets, 25% old exposure/contrast faults, and 50% veil faults, with equal update budgets against the existing synthetic schedule. Treat this as one data-distribution ablation. Combine faults only after single-fault behavior works. Store corruption seed per scene and variant, rather than deriving validation corruption from batch position, so changing batch size does not change the benchmark.

Flare7K++ provides public real and synthetic flare layers, but it is predominantly a **nighttime** benchmark. Its authors describe compositing layers onto clean images. It is a useful later diversity source, not a direct substitute for testing daytime backlit portraits; downloading it is unnecessary for the first experiment. [Official dataset and implementation](https://github.com/ykdai/Flare7K).

## 3. Correct no-op, highlight, and counterfactual supervision

1. Add full-pipeline same-style target-to-target examples: `G(y, style(y)) -> y`, including the global LUT. Keep their coefficient small initially because expert renditions are not guaranteed to be fixed points of a learned style operator. Monitor whether the model collapses toward universal identity on real source-to-target pairs.
2. On synthetic pairs, supervise both `G(corrupt(y)) -> y` on available pixels and `G(y) -> y`. A bounded output-difference consistency term should not replace the clean target: an identity model also satisfies some consistency constructions.
3. Preserve outside-support pixels relative to the known clean target for synthetic repair; use the detached clean grade only when the task is to maintain the model's same-style output under an input perturbation. Report these as distinct experiments.
4. Add a target-aware highlight loss on bright, low-chroma **target** pixels, separately measuring luminance and chroma errors. Define the threshold before the run and mask invalid/clipped-recovery pixels. An input-only rule that forces every white-looking area to remain white would preserve veil and block intended edits. Do not globally force skin, shirts, skies, and light sources to one brightness.
5. If local contrast regularly reaches its limit, first report that saturation rate and output gradient errors. Test a small penalty on contrast magnitude or excess beyond a declared soft range separately from map smoothness. A lower hard limit also changes representable edits; it needs a matched control.

The remaining gamut penalty is useful because it operates before final clipping. Supplement it with **new** clipping relative to the target where appropriate, not a blanket ban on legitimate pure white. Never infer that clipped JPEG texture has been recovered because a loss metric improved.

## 4. Perceptual and structure losses are second-line experiments

Keep RGB L1 as the reference objective. Once shape/data behavior is sane, separately try (a) small MS-SSIM plus RGB L1, or (b) a multiscale luminance-gradient matching term. Both should be target anchored; maximizing local contrast itself rewards exactly the overshoot being investigated. Zhao et al. show that loss selection can change restoration behavior with a fixed network, but their experiments do not identify the best weights for this grader. [Primary publication](https://research.nvidia.com/publication/2017-03_loss-functions-image-restoration-neural-networks).

LPIPS can be a secondary validation metric and a later weak feature loss. It was developed against human judgments of perceptual similarity, not specifically user preference for exposure or skin tone. A lower LPIPS does not eliminate the need for a side-by-side editing preference check. [Authors' implementation](https://github.com/richzhang/PerceptualSimilarity).

Avoid a universal mean-brightness target or gray-world color penalty as the first response to this photo. Zero-DCE uses a 0.6 exposure target and a color-constancy prior in its low-light setting; those priors can conflict with intentional bright backlight, white clothing, sunset color, or low-key scenes. Its useful lesson here is explicit curve constraints, not importing its aesthetic targets unchanged. [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf).

## 5. Cheap staged experiment plan

All budgets below are **optimizer updates**, not epochs or microbatches. Start with Expert C only for a clear test; A/B have just 17 training scenes each. Keep CNN, renderer, data order, effective batch size, initialization, and total source-image presentations matched. Benchmark actual local hardware before assigning wall-time or cost.

| Stage | Runs and budget | Decision it resolves |
|---|---|---|
| A: no training | Fixed ramps, gradients, gray/white patches; render baseline with global-only, local-only, full grade | Is the bad response already present in the executable recipe? |
| B: shape-only feasibility | Optimize LUT copies on 256 **training** scenes for 200 updates, encoder/local maps frozen; baseline no-update and matched plain-L1 continuation controls | Can a better-behaved LUT remove reversals without large reconstruction damage? This is a development probe, not generalization evidence. |
| C: scratch loss comparison | Two new global models, 600 updates each on the same 512-scene training subset: current losses vs normalized shape losses | Does shape regularization work during learning rather than only after training? |
| D: synthetic repair comparison | Freeze one clean scratch global model. Two matched local branches, 600 updates each: old faults vs additive-veil mix | Can current local operators learn the required repair? Measure a directly optimized recipe oracle on training pairs if both fail. |
| E: operator control, only if needed | 600 updates with identical data/loss for local affine/additive repair or the proposed tone operator | Does representational capacity, rather than data alone, explain the plateau? |
| F: conservative joint training | Only the best one or two candidates, 1,000 updates; separate full-pipeline no-op ablation | Do gains survive real paired grading without drift on acceptable images? |

Scale the winning intervention to the full new training partition, repeat at least two seeds, and use one longer matched continuation to check undertraining before considering a large network. Exact update budgets and starting weights must be recorded. A LUT-only repair of an old checkpoint cannot be compared to a scratch method as an independent validation result.

## Validation and evidence boundaries

Freeze a new development split **only inside** the original 1,924 training scenes: for example, 1,732 new-train and 192 development scenes, selected deterministically by scene ID/hash. All styles and variants of one scene stay together. Save a new registry and manifest rather than editing the frozen 70/30 files. The baseline for causal comparison must be retrained from scratch on the same 1,732 scenes; the old checkpoint has seen all 192 development scenes and is only a historical illustration.

Leave the 824-scene outer test closed during design, tuning, model selection, and threshold setting. After the candidate and reporting protocol are frozen, one final comparison can use it with transparent acknowledgement of its earlier historical evaluation. A truly fresh external set remains desirable for a stronger new generalization claim. Do not use the existing test results to pick a backlight slice or favorable examples.

On the new development partition, report:

- Overall Expert C PSNR, SSIM, RGB L1, and CIE76, including paired scene-level confidence intervals.
- Predefined bright/low-chroma target regions, shadow regions, and manually labeled backlit cases; show region sample counts and worst-tail errors, not only means.
- Fixed synthetic veil recovery on available pixels, newly clipped area, outside-support drift, and full-pipeline no-op drift. Use both unseen parameter values and a held-out field family.
- LUT negative slope rate **and magnitude**, curvature, chroma drift of neutral ramps, and local contrast/exposure saturation fractions. Inspect unclamped and exported 8-bit results.
- A fixed full-resolution visual sheet of 24–40 development scenes, including bright whites, faces, fine gradients, and already satisfactory photos. Blind and randomize candidate labels; allow “both unacceptable” and record reasons.

Freeze selection rules before comparison: reject candidates with new visible reversals/halos or consistent no-op damage; among eligible candidates compare paired restoration and visual preference. A small metric regression can be acceptable for a visible quality improvement, but it must be reported rather than hidden behind a composite score.

The supplied photograph can remain a clearly labeled failure illustration. It supplies neither the desired target grade nor a representative estimate of user preference. Collecting several preferred edits or blind choices would clarify whether “better” means less flare, brighter subject, darker foreground, warmer mood, or preserved backlight. That additional information is useful later; it is not required to fix the objective training gaps above.
