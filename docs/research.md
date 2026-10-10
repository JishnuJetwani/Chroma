# Selective local photo grading: research draft

Status: implementation and experiment protocol, not a completed empirical paper. Numbers belong in the measured-results section only after their checkpoint, source manifest, split and evaluation report have been recorded. A pilot proves that the pipeline runs; it does not establish novelty or state-of-the-art quality.

## Research question

Can a compact, style-conditioned grader improve exposure and contrast where needed while producing less unintended change elsewhere than equally budgeted global and spatial baselines? The intended contribution is a measured correction–preservation tradeoff under a local-compute constraint. It is **not** the invention of spatial enhancement, editable neural filters, or multiple neural stages.

The product accepts JPEG/PNG, predicts editable global color and local tonal controls from a small preview, and applies those controls to the original-resolution image. It does not generate replacement objects or recover information irreversibly clipped from the input. Training and development run locally on an M4 with 24 GB unified memory.

## Related work and what it rules out

| Prior work | Established capability | Consequence for this project |
|---|---|---|
| [HDRNet, 2017](https://groups.csail.mit.edu/graphics/hdrnet/) | CNN predicts a bilateral grid of local affine color transforms with edge-preserving full-resolution application. | A spatial CNN or an efficient low-resolution analysis branch is not new. |
| [Image-adaptive 3D LUTs, 2020](https://arxiv.org/abs/2009.14468) | Learns color tables and image-dependent mixing. | The global grader is a baseline and a useful module. |
| [DeepLPF, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Moran_DeepLPF_Deep_Local_Parametric_Filters_for_Image_Enhancement_CVPR_2020_paper.html) | Learns spatial polynomial, graduated and elliptical filters. | Interpretable local filters and sequential filter heads already exist. |
| [Spatial-aware 3D LUTs, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.html) | Uses global scenario weights and pixelwise fusion maps. | Spatial LUT mixing alone is not a novelty claim. |
| [MonetGPT, SIGGRAPH 2025](https://arxiv.org/abs/2505.06176) | Learns procedural retouching and reasoning from visual puzzles derived from edited photos. | Synthetic edit curricula and explainable edit sequences are also established. |
| [Modular Neural ISP, preprint 2025, revised 2026](https://arxiv.org/abs/2512.08564) | Modular editable rendering, multiple styles, and compact network variants. | A modular multi-style system requires a more specific empirical distinction. |

The repository's `bilateral` architecture is an internal locally affine baseline, not a claimed HDRNet reproduction. Its `unet` architecture predicts a low-resolution RGB residual; it is not an unconstrained full-resolution U-Net benchmark. External published checkpoints must use a pinned checkout and separate environment, with training-set overlap audited before reporting comparisons. The official [Modular Neural ISP repository](https://github.com/SamsungLabs/modular_neural_isp) currently documents MPS inference but not MPS training; adopting it directly would not satisfy the local-training requirement without additional work.

## Data and the curriculum

**FiveK establishes general grading.** [MIT–Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) contains 5,000 source photographs and five expert renditions. Begin with Expert C. When multiple editors become available, keep an explicit editor/style identifier and all versions of each photo in one scene split. These experts are editing preferences, not named cinematic presets. A downloaded preprocessed subset must be reported as that subset with its provenance; it is not automatically the full official benchmark.

**S24 supplies real locally adjusted renditions and preset styles.** The [S24 dataset description](https://raw.githubusercontent.com/SamsungLabs/time-aware-awb/main/s24-raw-srgb-dataset/README.md) documents 3,224 scenes with camera sRGB output, a manually edited Lightroom rendition with local tone mapping, and five additional preset renditions. The `masks` directory marks secondary illuminants, **not the artist's local edit masks or defective areas**. Privacy `blur_masks` must be excluded from loss/metrics. Camera sRGB and raw-derived renditions have a documented small spatial shift; audit alignment and use conservative valid masks before applying pixel losses. This single-camera corpus cannot alone establish broad JPEG generalization.

**Known perturbations teach restraint.** Apply recoverable local exposure/contrast perturbations to training-only images, varying soft support geometry and including unchanged examples. Single-family and mixed-family holdouts test whether the model learned an operation rather than an ellipse generator. Synthetic controls complement real S24 targets, not replace them. [Afifi et al.'s exposure dataset](https://github.com/mahmoudnafifi/Exposure_Correction) offers realistic rendered exposure variation but derives from FiveK; harmonize source IDs before mixing it with FiveK. A separately random split would leak scenes.

The default curriculum is: global Expert C pilot → repair-only perturbation pilot with the global branch bypassed → joint training on real paired images plus controlled perturbations and unchanged repair examples → S24 aligned real-pair/style training once acquired and audited. The S24 importer avoids shifted camera JPEGs by rendering supplied denoised raw RGB using camera white balance, the camera matrix and the standard sRGB transfer; this recipe must be recorded and visually aligned against targets. Keep FiveK and S24 validation reports separate. Use the actual manifest's style registry; do not invent names for S24 preset numbers or imply that untrained embeddings are usable styles.

The implemented joint schedule includes three real-pair batches followed by one control batch, with periodic repair-only identity examples. It includes reconstruction, map/LUT regularization, gamut and outside-region preservation terms. Operator-label auxiliary supervision, calibrated defect detection and a human local-annotation benchmark remain research extensions until explicitly implemented and tested.

## Model and hypotheses

The default `local` system shares one convolutional encoder between a style-conditioned global LUT mixer and a small spatial decoder. The decoder predicts exposure and log-contrast maps, upsampled using image guidance. Global-first rendering is the current default; local-first is an explicit ablation. The `cascade` comparator gives separate stages separate feature extractors, allowing us to test whether additional sequential analysis improves quality enough to justify its compute.

H1: A spatial branch reduces controlled region error relative to a global-only LUT. H2: Outside-region preservation training reduces unintended counterfactual drift at comparable correction quality. H3: Guided upsampling improves boundary behavior at full resolution compared with bilinear maps. H4: restricted highlight/shadow heads offer a more useful correction–preservation tradeoff than unrestricted maps. H5: the shared encoder is preferable to a cascade under a matched local training budget.

Map visualizations show what operation was executed. They do **not** prove why the model chose it. Final-image loss cannot uniquely identify a decomposition: multiple heads can duplicate or cancel each other. The optional tonal specialists constrain signs and brightness ranges, but this is not semantic defect supervision. A paper that claims specialized diagnosis additionally needs per-operation synthetic supervision retained during fine-tuning, intervention tests, and human fault labels. A bright pixel is not necessarily a mistake: snow, lights, reflective objects and intentional high-key images are negative controls.

## Evaluation and publication gates

Use the same scene splits, source rendering, effective batch size, optimizer-update budget and input resolutions for internal architecture comparisons. Record parameter count, synchronized training throughput, process/Metal allocation, and full-resolution rendering time. Select on validation; keep the test set unopened for model selection. For the main comparison use three training seeds after the pilot gate, not three seeds for every exploratory variant.

Paired metrics are PSNR, SSIM and D65 Lab **CIE76**, with exact definitions in each generated report. These measure target agreement, not universal aesthetics. [LPIPS](https://richzhang.github.io/PerceptualSimilarity/) is an optional supplementary perceptual-similarity metric, currently not required or reported by this implementation.

Counterfactual locality compares `F(clean, style)` and `F(corrupted, style)` outside the perturbed region, with a dilated exclusion band. Comparing the output against the raw input outside that region would incorrectly penalize the legitimate global grade. Report recovery inside, outside drift, boundary drift, and introduced clipping separately. Boundary drift is an inspection aid, not a validated halo score. Inspect hair, foliage, windows, clipped regions and smooth skies at original resolution. No-op scores require explicitly accepted images or the repair-only branch.

For the final candidate, collect blind comparisons on at least 100 independent held-out photos, ideally three raters each. Randomize display order and A/B assignment. Ask overall preference, correction quality and artifacts separately. Report ties. Average ratings within image, then scene, and bootstrap scenes; intervals condition on the recruited raters. Proposed success target: at least 60% decisive preference over the global baseline and a 95% scene-bootstrap interval above 50%, without a material no-op/artifact regression. This is a project decision rule, not a promise or established literature threshold. A statistically unclear result remains a valid negative result.

## Measured results

No publication results are asserted in this draft. Authoritative pilot outputs live in the corresponding `runs/<experiment>/` directory, including resolved configuration, checkpoint provenance, training history and evaluation JSON/CSV. Populate a results table only from those artifacts, label subset/short-run measurements as pilots, and distinguish synthetic controls from genuine human edits. Current human preference results: **not collected**. External baseline reproductions: **not established**. Novelty claim: **unproven pending comparisons and a final literature check**.

An external DeepLPF checkpoint smoke has run successfully on four pilot images using an isolated CPU environment. `runs/deeplpf_external_smoke/` records the pinned source commit, checkpoint hash, recovered-list overlap audit, per-image output/metrics, visual comparisons and a separate CPU benchmark. These inputs are roughly 720 by 480 pixels, with no additional resize in the external pass; this is neither a full-dataset reproduction nor a directly comparable result to internally trained 256-pixel pilots. All four IDs occur in the author's recovered test list, but that is provenance evidence rather than independent certification of the checkpoint's training history.
