# Measured results

This report distinguishes implementation checks from model-quality evidence. It will be updated with completed pilot measurements; no unfinished run is a result.

## Hardware and training feasibility

- Apple M4 MacBook Air, 24 GB memory; Python 3.11.14, PyTorch 2.8.0, MPS.
- Main model: 1,711,026 trainable parameters.
- 200 measured training steps after five warmup steps; batch two, 256-pixel CNN preview, 512-pixel padded render/loss.
- Mean step: **0.923 seconds**, median **0.792 seconds**, including loading, forward, backward and optimizer update.
- Metal driver allocated memory at end: **1,347,665,920 bytes** (not a peak-memory measurement).
- This measures a mixed joint-training workload on the small real-photo manifest, not a full-run ETA or original-resolution export latency. Other workloads were running on this Mac, so timing is indicative.

Raw record: `runs/benchmark_m4/benchmark.json`.

## Pipeline learning check

A deliberately small model (64-pixel preview, 17³ LUTs) trained on 16 actual FiveK photo pairs for 20 passes. Mean training loss fell from **0.1132 to 0.0493**, with validation L1 falling from **0.1278 to 0.0817** on four separate scenes. This is a gradient/data-pipeline check, not a quality benchmark or publication comparison. It should not be used to initialize a different split without checking scene overlap.

Raw records: `runs/overfit16/config.json`, `history.json`, `best.pt`.

## Completed 256-pixel pilots

Training uses the frozen 192-scene manifest: 128 training, 32 validation and 32 test scenes. All models use seed 42; checkpoints are selected on validation L1. CPU evaluation uses batch one, encoded sRGB, SSIM v2 with fully valid window support, and corruption version `local-recoverable-v2-headroom`. The architecture retains its planned 256-pixel encoder, 64-pixel maps and 33³ LUTs.

| Model | Allocated updates, including pretraining | Test PSNR ↑ | SSIM ↑ | CIE76 ↓ | Outside drift ↓ |
|---|---:|---:|---:|---:|---:|
| Fixed LUT | 160 | 17.9742 | 0.806807 | 16.7206 | 0 |
| Global CNN + LUT, initial | 160 | 18.1768 | 0.813748 | 16.2530 | 0.0000000203 |
| Global CNN + LUT, continued | 400 | 18.9014 | 0.830520 | 14.5765 | 0.000000000876 |
| Shared local model, preservation | 400 | 19.9172 | 0.857706 | 12.9404 | 0.00145317 |
| Shared local model, no preservation penalty | 400 | 20.1196 | 0.859866 | 12.7352 | 0.00212794 |

The completed global continuation adds **240 new updates** to the initial 160-update global checkpoint, matching the **400 allocated updates** of each local curriculum (160 global + 80 repair + 160 joint). Initialization hashes were verified. This is a match of allocated optimizer updates, not of FLOPs, wall time, curriculum or parameter count: global has 1,447,248 parameters, local 1,711,026. Validation selected the repair checkpoint after 48 of 80 updates and the local checkpoints after 144 or 160 joint updates. Actual selected-checkpoint lineages therefore contain **400 global, 352 local-with-preservation and 368 local-without-preservation updates**. The initial 160-update global row remains context only.

Compared with the completed 400-update global reference, both local variants improve clean paired-target agreement:

| Candidate minus continued global | PSNR difference, 95% CI | SSIM difference, 95% CI | CIE76 difference, 95% CI |
|---|---:|---:|---:|
| Local with preservation | +1.016 dB [0.434, 1.557] | +0.02719 [0.01980, 0.03482] | −1.636 [−2.163, −1.084] |
| Local without preservation | +1.218 dB [0.592, 1.819] | +0.02935 [0.02103, 0.03784] | −1.841 [−2.429, −1.213] |

The local variants also have lower target error in corrupted regions, but the **added error caused by corruption** does not clearly improve: its local-minus-global intervals include zero for both variants. Better overall target agreement cannot establish better defect recovery. Global outside drift is nearly zero; this alone does not establish selective correction, since a constant global transform also has zero drift outside a local perturbation. Both local variants increase outside and boundary drift relative to this global reference.

There is also a measured highlight tradeoff. **Local with preservation minus continued global** increases the fraction of newly high-clipped pixels by **0.427 percentage points**, 95% interval **[0.210, 0.687]**, and the all-channels-white fraction by **1.185 percentage points**. Boundary drift rises by **0.002300 MAE**, interval **[0.001338, 0.003348]**. Clipping counts use rendered encoded-sRGB channels at full scale (`>=1`), before output quantization; “new” excludes pixels already high-clipped in the input. These diagnostics cannot distinguish intentional whites from undesirable clipping or prove loss/recovery of detail. Nevertheless, the PSNR and regional-target-error improvements do **not** establish that the artifact gate has passed. No model is tuned on these test results.

The initial global checkpoint produced nearly constant photo-dependent mixing weights, so its small advantage over a fixed LUT does not establish effective adaptation. A controlled diagnostic shows that the architecture can learn conditioning; a context/diverse-basis alternative is implemented but unrun on real photos. These comparisons evaluate the complete trained systems and their curricula, rather than isolating the causal effect of adding a spatial head.

The preservation comparison holds architecture, training scenes, curriculum allocation and initialization fixed. Preservation reduces mean outside drift by **31.7%**, an absolute **0.000675 encoded-sRGB MAE** (about 0.172 of one 8-bit level). The paired scene-bootstrap 95% interval for the reduction is **[0.000281, 0.001102]**. It also lowers clean-image target agreement by **0.202 dB PSNR**, interval **[0.034, 0.360]**. Region recovery, added corruption error and boundary-drift differences have intervals crossing zero. This is evidence of a measured correction–preservation tradeoff, not of better recovery or perceptual quality. Intervals describe these 32 scenes and one training seed, without multiplicity correction.

Reports: `runs/{fixed,global,global_matched,local,local_no_preservation}_pilot/test/`. Paired comparisons and detailed interpretation: `runs/global_local_matched_comparison/`, `runs/preservation_comparison/` and `runs/fixed_global_matched_test/`. The matched comparison includes `training_lineage_audit.json`. Tests use a held-out edge-aware corruption family with combinations; initial training uses single faults from the other families. Optional combined-fault curriculum and exact exposure-inverse supervision are implemented as separate, unrun experiments.

## Repair-only abstention diagnostic

Both frozen local checkpoints also processed the **already edited Expert C targets** for the same 32 test scenes. Global grading was bypassed during analysis and rendering (`bypass_global=True`, grade strength zero, local strength one). This measures whether the repair head leaves a designated clean target unchanged; it is separate from full-grader identity or aesthetic preference. Evaluation used 256-pixel inputs, CPU with two threads, and the same validity masks.

| Repair head | No-op MAE ↓ | Valid pixels changed by more than 2/255 in any channel ↓ |
|---|---:|---:|
| With preservation | 0.0220640 | 70.02% |
| Without preservation | 0.0168046 | 69.60% |

The **without-minus-with** paired MAE difference is **−0.0052594**, with a 95% scene-bootstrap interval of **[−0.0074551, −0.0029131]**. The changed-pixel difference is −0.42 percentage points, interval [−7.72, 8.24]. Thus the preservation model changes these edited targets more in average magnitude, while the changed-pixel fractions are not clearly different. Average absolute channel changes correspond to about 5.63 and 4.29 levels on an 8-bit scale. Neither checkpoint demonstrates reliable abstention under this diagnostic.

This negative result does not contradict reduced outside-fault drift: the earlier metric compares corrupted-input output with clean-input output, and can remain small even when both outputs change the image. **Stability under local perturbations and leaving an already edited image alone are distinct properties.** Identity examples remain present in both training variants; the existing curriculum and preservation penalty have not established a general “no repair needed” behavior. These checkpoints were not tuned after this test diagnostic.

Five separate 64-pixel black, white, midgray, ramp and step fixtures produced finite outputs. Both models altered every pixel of the white fixture by more than 2/255; exact black preservation can follow from the multiplicative renderer. These are identity-policy stress checks, not artistic ground truth or independent photographic observations.

The diagnostic retains the same single-seed and 32-scene limitations, with 2,000 paired bootstrap resamples and no multiplicity correction. Source-code, target-image and checkpoint hashes, per-scene measurements, synthetic-fixture results and a replay script are in `runs/repair_noop_diagnostic/`; the original paired-evaluation reports are unchanged.

## External pretrained DeepLPF reference

The pinned author-repository DeepLPF checkpoint was evaluated on the **same 32 scenes, 256-pixel padded input tensors and valid-mask metric-v2 protocol**. All five input-side metrics match the internal reports exactly on every scene. All 32 IDs are present in its recovered 498-image test list and absent from its training/validation lists. These lists were recovered from a third-party mirror; membership was verified, but checkpoint training history was not independently certified.

| External checkpoint metric | Mean | 95% scene-bootstrap CI |
|---|---:|---:|
| PSNR ↑ | 24.8813 dB | [23.4658, 26.2544] |
| SSIM ↑ | 0.931890 | [0.915849, 0.946234] |
| CIE76 ↓ | 8.24634 | [7.16772, 9.46814] |
| Outside drift ↓ | 0.0108331 | [0.0082711, 0.0139580] |

**Training is unmatched:** the repository reports a 2,250-image training split and validation-peak checkpoint at epoch 774; our pilots use 128 training scenes and the budgets above. This is an external pretrained reference, not a training reproduction or an isolated architecture comparison. Its higher target agreement indicates substantial room for the pilot to improve, while its larger counterfactual drift reinforces that drift alone cannot rank enhancement quality. No human preference conclusion follows.

Checkpoint, source, split-list and output hashes; the exact-input audit; scored metrics; and inspectable input/output/target/mask PNGs are in `runs/deeplpf_external_32/`. The earlier four-image native-resolution smoke remains separately documented in `runs/deeplpf_external_smoke/` and is not pooled into these results. No latency conclusion is drawn from the loaded shared host.

## Three-editor application pilot

The final app checkpoint at `runs/multistyle_pilot/best.pt` contains **1,711,058 parameters** and styles **0 = FiveK Expert C, 1 = FiveK Expert A, 2 = FiveK Expert B**. It initialized from the selected Expert C local checkpoint, transferring the C embedding by name and initializing A/B independently. The extension trained for 20 epochs/120 allocated updates; validation selected epoch 19/update 114. Training used 16 scenes with three targets each, validation four scenes, and testing four separate scenes.

| Test style (four scenes each) | Target MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| C | 0.07214 | 21.92 | 0.8871 |
| A | 0.07983 | 20.80 | 0.9014 |
| B | 0.05480 | 23.66 | 0.9407 |

The twelve target renditions are **four independent scenes**, not twelve independent photos. These results are separate from the single-style model comparison. Changing the style on identical inputs produces non-identical outputs, but pairwise differences average only **0.78–2.12 RGB levels out of 255**, much smaller than the corresponding human-target differences. Conditioning exists but is weak; this does not establish successful reproduction of the editors' preferences. The names identify training editors, not film aesthetics.

Checkpoint/initialization hashes, named-embedding transfer and registry order were verified. Metrics and style-conditioning diagnostics: `runs/multistyle_pilot/test/`. The live three-style MPS app passed the API smoke in `runs/app_smoke_multistyle.json`.

## Native-resolution application benchmark

The final three-editor checkpoint (1,711,058 parameters) ran on MPS using a **2832 × 4256 image (12.05 megapixels)**. This is a native-resolution Expert A rendition used only as a performance fixture, not an input/target quality test. One cold CNN analysis took **3.108 seconds**. After one preview warmup, three cached 681 × 1024 renders took **0.124 / 0.247 / 0.271 seconds** (median **0.247 seconds**).

Three original-resolution PNG exports took **13.149 / 8.624 / 8.753 seconds**, median **8.753 seconds**, including source tensor conversion, tiled rendering and PNG encoding. They exclude initial file decoding and checkpoint loading. The first export exceeds the ten-second target, so this is not a latency guarantee. Source dimensions and checksum were preserved. Other host workloads were present; photo-grader training had finished before this run.

Metal driver allocation at the end was **1,244,643,328 bytes**, not peak memory. Tiling bounds processing scratch but several full-frame RGB buffers remain during assembly and encoding. The implementation has tile/full-render equality tests; a scaled visual inspection of this export showed no obvious grid seams, but does not certify absence of halos or banding across photos.

Raw timing, hashes and three exported PNGs: `runs/inference_12mp/benchmark.json`. The application smoke checks original-size export, all three style choices, exact zero-strength pixel bypass, clearing obsolete downloads and preserving source bytes. The application has automated export, bypass and frozen-split reconstruction checks.

## Claims not established

- Novelty or state-of-the-art performance.
- Human preference superiority or preservation of every intentional photographic effect.
- Reliable repair-head abstention on already edited photographs; the completed no-op diagnostic shows substantial changes.
- Generalization of local correction to genuine locally edited targets (S24 not yet acquired).
- Three-seed significance, learned human multi-style quality, or a reproduced DeepLPF training result.
- Under-one-second preview or under-ten-second 12-megapixel export guarantees.
