# A pilot study of selective local photo grading on Apple Silicon

**Experimental report, October 2026.** This is a single-seed pilot on a small, documented FiveK subset. It makes no novelty, state-of-the-art or human-preference claim. The accompanying implementation and source artifacts support each reported measurement.

## Abstract

We implement a compact photo grader that predicts a global color lookup table and spatial exposure and contrast adjustments. The network analyzes a 256-pixel preview; a differentiable renderer applies one predicted recipe at the photograph's original resolution. On 32 held-out scenes at 400 allocated updates per system, the local model with preservation improves paired-target PSNR over a continued global baseline by 1.016 dB, while differing in curriculum and parameter count. A controlled experiment tests whether an outside-region consistency penalty reduces changes induced by local synthetic faults. Adding the penalty reduces mean outside-region drift from 0.002128 to 0.001453 in encoded-sRGB mean absolute error, a 31.7% relative reduction, while costing 0.202 dB of clean paired-target PSNR. Region-recovery and boundary-drift differences remain statistically unclear. A separate repair-only test on already edited targets finds substantial changes from both variants, with greater average change for the preservation model (MAE 0.02206 versus 0.01680). Reduced counterfactual drift therefore does not establish abstention. An external pretrained DeepLPF checkpoint achieves higher target agreement under the same evaluation inputs, with unmatched training. The pilot does not establish perceptual benefit, generalization to genuine local edits, or robustness across training seeds.

## 1. Motivation and relation to existing methods

Automatic grading should produce a coherent treatment while avoiding unnecessary changes when only part of a photograph is perturbed. Spatial adjustment is established prior art: [HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/) predicts bilateral transforms, [DeepLPF](https://openaccess.thecvf.com/content_CVPR_2020/html/Moran_DeepLPF_Deep_Local_Parametric_Filters_for_Image_Enhancement_CVPR_2020_paper.html) predicts local photographic filters, and [RSFNet](https://arxiv.org/abs/2303.08682) uses regional masks and operations. [Image-adaptive 3D LUTs](https://arxiv.org/abs/2009.14468) supply a precedent for efficient global grading. [CoTF](https://github.com/HUST-IAL/CoTF) and [Modular Neural ISP](https://arxiv.org/abs/2512.08564) further limit architectural novelty claims. Synthetic retouching curricula are also established, including in [MonetGPT](https://arxiv.org/abs/2505.06176).

Our pilot asks a narrower empirical question: **does explicit consistency outside a known perturbation reduce counterfactual output drift in a small, editable grading model?** A bright area is not automatically a defect. Predicted maps represent adjustment amounts, not calibrated defect probabilities.

## 2. System

The main model has 1,711,026 trainable parameters. Four convolutional stages use 32, 64, 128 and 128 channels, with a 16-dimensional style embedding. A global head mixes eight learned 33³ RGB lookup tables. A small spatial decoder predicts exposure and log-contrast maps on a 64-pixel canvas, with skip connections and globally graded luminance as additional input. Aspect ratios are preserved using padding; prediction maps are cropped into normalized source coordinates.

Exposure is bounded to ±1 stop. Contrast is represented in log₂ space and bounded to effective multipliers 0.5–2.0. The renderer converts sRGB into linear RGB, decomposes log luminance into a guided smooth base and local detail, and adds exposure while scaling detail. Ratios are applied to linear RGB to preserve chromatic relationships. Guided upsampling follows image boundaries; bilinear upsampling is an explicit alternative. LUT interpolation uses eight-corner gathers to support MPS gradients without custom CUDA code.

The application separates `analyze(preview, style)` from `render(original, recipe, strengths)`. Cached strength changes require no new CNN analysis. Large-image rendering reuses one whole-image recipe across overlapping tiles; zero strength is an exact image bypass. The system exports 8-bit sRGB JPEG/PNG and preserves the original file. It cannot reconstruct information absent from clipped pixels.

Internal fixed-LUT, adaptive-global, cascaded-specialist, RGB-residual and bilateral-grid alternatives are implemented. Their parameter counts differ and are reported explicitly. They are not reproductions of the named external methods.

## 3. Data and protocol

The [MIT–Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) source contains 5,000 photographs with expert renditions. We acquired 2,748 recoverable input/Expert-C pairs from the documented prepared release and its supplied split lists, preserving 498 published test IDs and deriving validation from training IDs. An immutable 192-scene subset supplies 128 training, 32 validation and 32 test scenes for this pilot. It is not the full official benchmark. Checksums, rendering versions, color-profile conversion and scene-disjoint splits are recorded in manifests; padding and privacy-invalid pixels are excluded from loss and metrics. Full details and limitations appear in [data preparation](data.md).

A separate 24-scene corpus contains real C/A/B targets, with native conversion of the original 16-bit ProPhoto A/B TIFFs before 8-bit sRGB quantization. A separate three-style extension was trained for 20 epochs (120 updates), selecting epoch 19 by validation. It does not enter the following single-style comparison. On four independent test scenes, style-specific PSNR ranged from 20.80 to 23.66 dB. Output differences between editor choices averaged only 0.78–2.12 levels out of 255, substantially smaller than human-target differences: conditioning was measurable but weak, and editor-style fidelity remains unestablished. The [S24 corpus](https://github.com/SamsungLabs/time-aware-awb/blob/main/s24-raw-srgb-dataset/README.md), which includes genuine manual local tone edits, has not been acquired. Its importer is fixture-tested; a real-data local-edit experiment remains outstanding.

Training uses PyTorch 2.8, FP32 MPS, AdamW with learning rate 0.0002 and weight decay 0.0001, microbatch two and gradient accumulation four. The initial fixed and global pilots allocate 160 optimizer updates each. A completed global continuation adds 240 updates from the initial global checkpoint, for 400 total. Both spatial branches share the same initialization and allocate 160 global, 80 repair and 160 joint updates. Validation L1 selects checkpoints within those allocations. The repair stage bypasses global grading and corrupts edited training targets. Joint training mixes real paired examples with synthetic faults and unchanged repair examples.

Initialization checkpoint hashes are verified. The continued global checkpoint contains 400 updates in its selected lineage. The selected repair checkpoint contains 48 of its 80 allocated updates, followed by 144 selected joint updates with preservation or 160 without; local selected lineages therefore contain 352 and 368 updates. The comparison matches **allocated optimizer updates**, not selected-checkpoint update count, FLOPs, wall time, curriculum or parameters (1,447,248 global versus 1,711,026 local).

Training faults use feathered ellipses, gradients and irregular regions. Evaluation holds out an image-boundary-aware generator and includes combinations. The corrected `local-recoverable-v2-headroom` generator avoids new clipping; superseded runs with the earlier contrast-fault generator are excluded. The main pilot uses single-fault training. Optional combined-fault curriculum and exact known exposure-inverse supervision are separate, implemented but unrun variants.

The controlled comparison toggles only the outside-consistency penalty, retaining identity examples and other losses. Consistency compares the model's outputs for the original and locally corrupted input at the same style, outside a dilated perturbation support. It does not penalize the intended global grade by comparing against an ungraded original.

## 4. Evaluation and results

All internal test reports below evaluate the same 32 scenes at 256 pixels on CPU with batch one and seed 42. Metrics are encoded-sRGB PSNR, SSIM with Gaussian 11/1.5 windows and fully valid support, and D65 Lab CIE76. Counterfactual locality reports errors inside and outside the known synthetic support separately. Confidence intervals use 2,000 paired scene-bootstrap resamples. They do not measure uncertainty across training seeds or recruited raters; multiple comparisons are not corrected.

| Model | Allocated updates | PSNR ↑ | SSIM ↑ | CIE76 ↓ | Outside drift ↓ |
|---|---:|---:|---:|---:|---:|
| Fixed LUT | 160 | 17.9742 | 0.806807 | 16.7206 | 0 |
| Adaptive global, initial | 160 | 18.1768 | 0.813748 | 16.2530 | 0.0000000203 |
| Adaptive global, continued | 400 | 18.9014 | 0.830520 | 14.5765 | 0.000000000876 |
| Shared local, preservation | 400 | 19.9172 | 0.857706 | 12.9404 | 0.00145317 |
| Shared local, no preservation penalty | 400 | 20.1196 | 0.859866 | 12.7352 | 0.00212794 |

The 160-update global row is historical context; the completed 400-update continuation is the allocated-budget reference. Relative to that reference, local with preservation improves PSNR by **1.016 dB**, paired 95% interval **[0.434, 1.557]**, SSIM by **0.02719** [0.01980, 0.03482], and CIE76 by **−1.636** [−2.163, −1.084]. Local without preservation improves PSNR by **1.218 dB** [0.592, 1.819], SSIM by **0.02935** [0.02103, 0.03784], and CIE76 by **−1.841** [−2.429, −1.213]. These compare complete trained systems and curricula; they do not isolate the causal effect of the spatial head.

Both local variants lower paired-target error within corrupted regions, but their differences from global in the **additional error caused by corruption** have intervals containing zero. This separates better general target fit from improved fault recovery. Both local variants increase outside and boundary drift relative to global. Near-zero global outside drift is insufficient evidence of selective repair: a constant global transform has this property by construction. The initial global pilot's mixing weights varied little across validation images; its small fixed/global advantage should not be interpreted as successful learned adaptation. Controlled diagnostics establish that the architecture can learn conditional behavior, but do not establish real-photo superiority for the optional context/diverse-basis variant.

Highlight diagnostics also show a tradeoff: **local with preservation minus continued global** increases newly high-clipped pixels by **0.427 percentage points**, paired 95% interval **[0.210, 0.687]**, and all-channels-white pixels by **1.185 percentage points**. Boundary drift increases by **0.002300 MAE** [0.001338, 0.003348]. Clipping is measured at rendered encoded-sRGB full scale (`>=1`) before output quantization; newly clipped pixels exclude those already high-clipped in the input. Intentional whites are possible, so these counts alone do not identify undesirable clipping or establish detail recovery. They do prevent treating improved PSNR or regional target fit as a passed artifact gate. No checkpoint is tuned in response to these test findings.

Within the matched spatial comparison, preservation lowers outside drift by **0.000675**, paired 95% interval **[0.000281, 0.001102]**, or **31.7%** relative to the no-preservation model. This absolute difference is approximately 0.172 of one 8-bit level on average, so its visual importance remains unknown. Clean paired PSNR is lower by **0.202 dB**, interval **[0.034, 0.360]**. Region-recovery, corruption-penalty and boundary-drift difference intervals cross zero. We therefore report reduced counterfactual drift with a small target-agreement cost, rather than improved recovery or aesthetics.

We separately test repair-head abstention by feeding the same held-out **human-edited targets** to each frozen model. Analysis bypasses the global LUT, and rendering sets global strength to zero and local strength to one. Validity masks exclude padding and privacy-invalid pixels; all other image-size, CPU and bootstrap settings remain as above. This diagnostic designates the edited target as an image needing no repair. It does not assert a unique optimal artistic rendition or test full-style-grader idempotence. No model was tuned using this diagnostic.

| Repair-only checkpoint | No-op MAE ↓ | Valid pixels changed by more than 2/255 in any channel ↓ |
|---|---:|---:|
| With preservation | 0.0220640 | 70.02% |
| Without preservation | 0.0168046 | 69.60% |

The paired difference, **without preservation minus with preservation**, is **−0.0052594 MAE**, 95% interval **[−0.0074551, −0.0029131]**. The corresponding changed-pixel difference is −0.42 percentage points, interval [−7.72, 8.24]. Both heads alter many pixels; the preservation variant has larger mean change, and neither demonstrates reliable abstention. This is compatible with reduced counterfactual outside drift, which compares two model outputs and does not require either to equal a clean target. Separate 64-pixel black/white/midgray/ramp/step stress fixtures remain finite but are not artistic ground truth or part of the scene-bootstrap sample.

A 200-step Mac training benchmark measured a mean 0.923 seconds per two-image step with 512-pixel rendering, including loading, forward/backward and optimizer work. Metal allocation at the end was approximately 1.35 GB; this is not peak memory. Other workloads were present. A separate final three-editor checkpoint benchmark used a 12.05-megapixel native-resolution fixture on MPS. Median cached preview rendering was 0.247 seconds; three PNG exports took 13.149, 8.624 and 8.753 seconds (median 8.753). Export timing includes tensor conversion, tiled rendering and encoding, but excludes initial decoding/model loading. Cold analysis took 3.108 seconds. End-of-run Metal allocation was 1.24 GB, not peak memory. These measurements come from a shared host and one fixture; the first export missed the ten-second target.

A separate **external pretrained DeepLPF reference** uses the same 32 scenes and 256-pixel input tensors, padding/validity masks and metric-v2 implementation. All input-side metrics match the internal reports exactly. All 32 IDs occur in the pinned repository's recovered DPE test list and none in its training/validation lists. The lists were recovered from a third-party mirror, and checkpoint training history is not independently certified.

| External pretrained reference | PSNR ↑ | SSIM ↑ | CIE76 ↓ | Outside drift ↓ |
|---|---:|---:|---:|---:|
| DeepLPF, recovered-DPE checkpoint | 24.8813 | 0.931890 | 8.24634 | 0.0108331 |

**Training is unmatched:** the repository reports a 2,250-image training split and checkpoint selection at the validation-PSNR peak at epoch 774. This is neither our training reproduction nor the original paper benchmark. The higher paired-target agreement supplies useful quality context and shows substantial room to improve the pilot; it does not isolate architectural superiority. Larger counterfactual drift also illustrates why that metric cannot alone rank enhancement quality. The earlier four-image native-resolution smoke is separate and is not pooled into this evaluation.

## 5. Discussion and limitations

The central positive observation is that a simple consistency penalty can reduce outside-region drift under controlled perturbations. The same experiment reveals a tradeoff in target reconstruction, and does not demonstrate better fault recovery. The repair-only diagnostic adds a negative finding: both variants modify already edited targets, and preservation increases the mean modification magnitude in this pilot. Stability under a local perturbation and recognizing that no repair is required must therefore be evaluated separately. Larger scene counts, multiple seeds, genuine locally edited data and blind human comparisons are required before selecting the method for quality rather than implementation convenience.

The FiveK source renditions and small subset limit transfer to ordinary camera JPEGs. Synthetic perturbations do not identify whether a real highlight, silhouette or contrast choice is undesirable. Head decompositions are not unique: multiple operations can partially cancel, so map visualization is not a semantic explanation. Tests of identity, gradients, color conversion and tiled rendering establish implementation properties, not absence of artifacts on all photographs.

The completed allocated-budget comparison supports improved target fit for the local system in this pilot, while the external reference exposes a remaining quality gap under unmatched training. The next empirical gates are larger-data and multiple-seed comparisons; S24 rendering/alignment validation; stronger supervision for recoverable exposure faults and abstention; and reserved personal photographs with blinded preference/artifact ratings. Any intervention prompted by this no-op result must be developed using training/validation data and assessed on a fresh reserved test set. A 60% decisive-preference rate is a project target, not a measured result. Until those gates are met, the application is labeled an early research preview.

## Reproducibility artifacts

- [Setup and app usage](../README.md)
- [Experiment configurations and exact replay protocol](experiments.md)
- [Verified data and provenance](data.md)
- [Measured results and raw-report locations](results.md)
- Local checkpoints and reports: `runs/*_pilot/`; paired preservation interpretation: `runs/preservation_comparison/interpretation.md`.
- Global/local paired differences and initialization/allocated-update audit: `runs/global_local_matched_comparison/`.
- External pretrained DeepLPF metrics, exact-input audit, hashes and inspectable inputs/outputs: `runs/deeplpf_external_32/`.
- Repair-only no-op protocol, per-scene records, source/target/checkpoint hashes and reproduction script: `runs/repair_noop_diagnostic/`.

The data retain their original research/non-commercial terms. No private user photographs or human ratings were used in the reported experiment.
