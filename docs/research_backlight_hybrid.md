# Hybrid inference for the backlit JPEG

Research direction 4, 2026-10-21. Scope: system composition, conservative operator search, diagnostics, selection, and available pretrained comparators. This is a proposed local inference experiment, not a trained or validated replacement. No application code or model files were changed; the photo was not uploaded.

## Recommendation

Keep the current network as a recipe proposer. Surround it with an original-image diagnostic pass, a small set of constrained candidate recipes, and explicit rejection rules. Always include identity. Protect clipped, textureless highlights from being interpreted as recoverable detail; let measured texture in the shirt and foreground justify mild local contrast. Do not make a no-reference quality score the optimizer or final authority.

This differs from retraining a stronger enhancer: it can test the immediate failure with the existing checkpoint, renderer, and installed DeepLPF baseline. It also reveals whether the bottleneck is a bad LUT, spatial correction, or missing expressive power.

## Evidence from this workspace and image

The source is a 640 × 360 JPEG. Visual inspection shows a large white sky, backlit hair, a white shirt with surviving folds, foreground vegetation under a broad light veil, and a localized pink flare. Keeping the luminous mood is a reasonable working intent; there is no ground-truth target establishing how much glow the user wants removed.

`runs/backlight_20261021/baseline_stats.json` reports 35.6576% white-clipped pixels in the source and 0% in the current model output. That is not evidence of recovery. The count can become zero merely by turning a constant white patch into a constant gray patch. Measure **new** clipping outside the original clipped region separately; do not reward reducing existing clipping.

Relevant implementation facts, read locally:

- `Grader.analyze` obtains one cached recipe; `render_preview` accepts independent global/local strengths. Candidate generation can reuse that recipe without repeated CNN inference.
- `EditRecipe` contains independent LUT, exposure, and log-contrast tensors. Exposure is ±1 stop; the contrast map controls a log-luminance detail multiplier. It is possible to disable one head while testing the other.
- `render_with_aux` exposes pre-clamp RGB. Use this for gamut diagnostics; a post-clamp PNG hides the magnitude of overshoot.
- LUT training uses a soft penalty for negative adjacent differences on all three lattice axes, not a hard constraint. The companion architecture investigation reported 43 negative per-channel increments out of 165 on the scene's neutral ramp from code 200 to 255, with a worst step around −0.009 for a +1/255 input step and maximum amplification around 4.94×. Those measurements were supplied by the coordinating agent, not independently remeasured here.

## Published support and its limits

**Operator-based decision making.** Hu et al.'s *Exposure* models retouching as successive choices of interpretable, resolution-independent filters, with feedback after each step. Its training uses reinforcement learning and a learned preference discriminator. The useful precedent here is the restricted action space and explicit stopping option; this proposal does not require reproducing its learning system or importing its old TensorFlow stack. [Author paper](https://people.csail.mit.edu/hehaodele/projects/TOG18-exposure/paper.pdf), [author implementation](https://github.com/yuanming-hu/exposure).

**A distinct local-filter comparator.** DeepLPF predicts elliptical, graduated, and polynomial filters. It is a suitable different parameterization to compare against the current dense exposure/contrast maps. Its paper's results do not establish performance on this already processed backlit JPEG. [CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Moran_DeepLPF_Deep_Local_Parametric_Filters_for_Image_Enhancement_CVPR_2020_paper.html).

**Metric optimization has failure modes.** Liu et al. demonstrate that imperceptible adversarial changes can substantially change learned no-reference IQA scores. This is evidence against treating score optimization as guaranteed perceptual improvement, not proof that every finite candidate ranking fails. [CVPR 2024 author paper](https://arxiv.org/abs/2403.11397).

**A fixed exposure target is not scene intent.** Zero-DCE uses a local exposure-control loss; its published setting is 0.6. Its official code also penalizes unequal mean color channels. On this image, applying those objectives indiscriminately would pressure an intentionally white sky toward middle gray and could suppress warm light. That implication is our scene-specific inference, not a claim that the authors intended those uses. [Author paper](https://staff.ie.cuhk.edu.hk/~ccloy/files/cvpr_2020_zerodce.pdf), [official loss implementation](https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/Myloss.py).

## Diagnostic and routing proposal

All masks and regions must be computed from the original and then frozen for candidate scoring. Otherwise a candidate can change which pixels are evaluated. The following numbers are starting values for experiments, not calibrated confidence thresholds.

| Diagnostic | Prototype definition | Consequence |
|---|---|---|
| Irrecoverable bright interior | All RGB channels ≥250/255 with 5×5 luminance standard deviation <2/255; separately record the component touching the top edge | Preserve these pixels; do not score texture gain or exposure improvement there |
| Highlight transition | Soft luminance ramp starting near 0.85 and ending near 0.98, plus distance to the bright interior | Feather correction down smoothly; inspect the transition for rings and halos |
| Recoverable bright texture | Bright pixels below full clipping with original gradients above the flat/noise floor | Allow mild detail adjustment while constraining mean luminance/chroma drift; white shirt folds belong here |
| Extreme backlight | Large bright-flat fraction plus substantial darker content; this source clearly qualifies | Restrict global strength; use foreground detail/veil candidates instead of uniform exposure reduction |
| Bad LUT response | Neutral and near-neutral ramps have negative slope, large positive slope, or abrupt chroma change | Reject full-strength global transform; offer identity blend or projected-LUT candidate |
| Weak texture / noise risk | Detail energy is close to flat-region noise estimate | No dehaze/clarity reward; cap gain and preserve low-frequency light |
| Recipe disagreement | Global-only, local-only, and reduced-strength outputs imply incompatible tonal directions | Treat as uncertainty, not votes; select a conservative candidate unless detail evidence is clear |

Brightness is not semantic sky detection. A white garment can enter the bright mask, and that is acceptable for protecting flat white areas. Only use a “sky” label if a separate semantic method actually establishes it. Likewise, a histogram cannot establish that a glow is unwanted. Keep the glow-preserving candidate in the comparison.

Use original luminance as the guide for mask refinement. Do not sharpen the mask with the candidate image. A hard threshold around the shirt creates a visible cutout; broad feathering alone can leak over edges. Test guided feathering and inspect the boundary at source resolution.

The first prototype can blend a guarded rendered candidate with the source:

`output = source + correction_weight(source) * (candidate - source)`

This is an experiment outside the application, so it need not immediately fit `EditRecipe`. Log the mask definition and strength alongside the recipe. For a final feature, preserve the same source-coordinate mask during full-resolution export. A final composite guard protects both global and local damage; guarding only the exposure map leaves LUT-induced contouring intact.

## LUT validity and bounded repair

Before searching local operators, run the scene LUT over a 256- or 1024-sample neutral ramp and small nearby color offsets. Record minimum finite difference, maximum gain, chroma variation, and any new extrema. A neutral ramp does not certify the whole RGB cube, but it targets this image's dominant near-white failure cheaply.

Two distinct experimental repairs are useful:

1. **Strength restriction.** For `T_s(x) = (1-s)x + s T(x)`, a ramp segment with minimum slope `m < 0` becomes nondecreasing when `s ≤ 1/(1-m)`. With the reported worst step, `m ≈ −2.30`, giving an upper bound near 0.30; use a margin, e.g. 0.25. Also cap positive slope: to keep gain at most `K`, require `s ≤ (K-1)/(M-1)` when maximum original gain `M > K`. With `M ≈ 4.94` and `K=2`, this is about 0.25. These bounds apply only to the sampled ramps and global transform. Spatial blending, local maps, and RGB off the tested curves still need inspection.
2. **Projected candidate LUT.** Copy the mixed scene LUT and impose nonnegative adjacent differences on each output channel along each of its three input axes. Repeated one-dimensional isotonic projections can find a feasible cube; use Dykstra corrections if claiming the closest Euclidean projection. Check convergence and all adjacent differences explicitly. Clamping projected values to [0,1] preserves this ordering. Render this copy as another candidate; never mutate the checkpoint. This matches the repository's intended all-axis monotonicity, but can change cross-channel color behavior, so also measure edit distance and inspect colored vegetation. Monotonicity does not enforce smooth slope, near-white neutrality, or correct photographic intent.

Do not call a single cumulative-maximum pass a minimal repair, and do not claim that fixing the neutral diagonal repairs the full cube. A heavy projection should trigger a fallback to lower strength rather than silently replacing the user's style. A later training fix can make constraints intrinsic; that is outside this inference-only proposal.

## Concrete bounded experiment with existing weights

Use one cached recipe and at most 15 source-resolution outputs:

1. Nine strength combinations: global `{0, 0.25, 0.5}` × local `{0, 0.25, 0.5}`, including identity. Retain the previously rendered full-strength model as a reference.
2. Two guarded variants of the best safe combination: protect clipped-flat interiors only, then protect interiors plus feathered bright transitions. Compare the effect on both sky and shirt.
3. One projected-LUT candidate with the same local setting. Include a synthetic ramp panel alongside the photograph.
4. Two conservative detail/veil candidates on the guarded best candidate, one mild and one medium. Let the classical-operator experiment supply the exact operator; keep its bounded local strength separate from exposure and disable it in the bright interior. Avoid a global airlight estimate that mistakes the white shirt or sun for diffuse haze.
5. One installed DeepLPF output as an independent comparator. If it is clearly worse, do not spend time fitting a blend to this single image.

The coordinating agent reports that stronger veil subtraction around 0.12 crushes the greens while mild subtraction helps. Keep that as observed scene-specific feedback; it does not establish a universal dehaze setting. Optical flare, atmospheric haze, and deliberately soft grading are not interchangeable image-formation models.

For each result, log pre-clamp overshoot, newly clipped fraction outside original clipping, luminance/chroma drift in the bright interior, original-edge retention, and source-defined region detail change. Review a fixed contact sheet containing the full frame and identical crops of sky transitions, hair outline, shirt folds, pink flare, and foreground. Preserve the original lighting impression as an explicit review criterion.

## Selection without pretending to know aesthetics

Use two stages:

- **Reject visible/structural damage first.** Reject newly introduced contours, large neutral-highlight tint, flat-sky dimming beyond a small tolerance, halo bands, destroyed shirt-fold edges, and significant new black or white clipping. Prototype warning levels could be 0.02 sRGB mean highlight shift and 0.1 percentage points of new clipping; neither threshold has been validated for arbitrary photos.
- **Rank only feasible candidates.** Reward modest detail gain in regions that already contain signal, with a saturating target band rather than unlimited sharpness. Penalize departure from the original outside justified regions. Prefer the smaller edit when gains are within the measurement uncertainty. Select identity if no candidate provides a clear gain.

Do not maximize entropy, variance, saturation, edge energy, or closeness to a global mean. Each can be improved by a visibly bad edit. Preservation-only scoring also fails because it always favors identity; it is a guard, not a complete objective. An optional pretrained NR-IQA score may be a secondary tie-breaker after constraints and should be reported separately, never combined into an unexplained “quality percentage.”

For automation beyond this example, use a held-out set of white clothing, snow, fog, deliberate silhouettes, sunsets, and ordinary well-exposed images. Record human pairwise preference and the frequency of unacceptable regressions. A single attractive result demonstrates a useful candidate, not a reliable router. Thresholds should be chosen from those failures, rather than fitted to make this photograph win.

## Available pretrained baselines and Mac feasibility

### Use first: installed DeepLPF

The workspace already contains:

- `third_party/deeplpf/pretrained_models/adobe_dpe/deeplpf_psnr_24.180_ssim_0.917_model.pt` — 6,903,041 bytes.
- `third_party/deeplpf/pretrained_models/adobe_dpe_recovered_split/deeplpf_psnr_23.833_ssim_0.916_model.pt` — 6,921,057 bytes.

Local `deeplpf_cli.py` implements CPU/MPS/CUDA selection and loads weights through CPU. Its `enhance_image` accepts this image's dimensions. Pin the explicit checkpoint because the default uses the first sorted glob match. For provenance, the local README describes these as repository retrainings on different splits, not the unchanged original CVPR checkpoint; their PSNR values are not comparable with this project's split.

Run the repository CLI from the existing environment into a new experiment directory, specifying `--device cpu` initially. Check MPS parity if MPS is used for actual output. The CLI reads via PIL directly, so for a fair pipeline comparison give it an sRGB/EXIF-normalized copy made by the current project's loader. This baseline requires no additional model download. [Author repository](https://github.com/sjmoran/deeplpf-image-enhancement).

### Optional later comparator: IAT exposure checkpoint

IAT is a small local/global enhancer targeting under- and over-exposure as well as low light. The author reports approximately 90K parameters; the advertised 0.004 s was measured on an NVIDIA 3090, not a Mac, and the README explicitly warns about single-image startup latency. [Author repository](https://github.com/cuiziteng/Illumination-Adaptive-Transformer), [BMVC paper](https://arxiv.org/abs/2205.14871).

The official GitHub tree lists `IAT_enhance/best_Epoch_exposure.pth` as 427,160 bytes, blob SHA `ff0c7072821c658b73abaa23b0ebc21884db2d3c` (metadata checked 2026-10-21). No weights were downloaded here. The demo has a separate exposure path; choose that checkpoint rather than its LOL low-light checkpoint. [Official exposure instructions](https://raw.githubusercontent.com/cuiziteng/Illumination-Adaptive-Transformer/main/IAT_enhance/readme.md), [repository tree metadata](https://api.github.com/repos/cuiziteng/Illumination-Adaptive-Transformer/git/trees/main?recursive=1).

The demo hardcodes `.cuda()` for model and input, so it is not a drop-in Mac command. A small local wrapper can load on CPU and avoid its evaluation-only imports. The forward path uses ordinary PyTorch convolutions, pointwise operations, `tensordot`, and gamma/color transforms; CPU feasibility is a code-based inference, and MPS support remains untested. `timm` imports must be version-checked. Use `torch.load(..., map_location='cpu', weights_only=True)` and validate exact state-dict compatibility; do not relax loading silently. [Official demo](https://raw.githubusercontent.com/cuiziteng/Illumination-Adaptive-Transformer/main/IAT_enhance/img_demo.py), [model implementation](https://raw.githubusercontent.com/cuiziteng/Illumination-Adaptive-Transformer/main/IAT_enhance/model/IAT_main.py), [global branch](https://raw.githubusercontent.com/cuiziteng/Illumination-Adaptive-Transformer/main/IAT_enhance/model/global_net.py).

It is a useful future comparator if the installed baseline and constrained recipe search are inadequate. It is not needed to complete today's experiment, and neither it nor DeepLPF can recreate actual scene detail absent from a clipped JPEG.
