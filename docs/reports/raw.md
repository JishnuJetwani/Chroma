# RAW pilot: implementation and measured results

2026-10-30. The local app now opens a separate RAW developer, using the shared global/local RAW checkpoint. JPEG grading remains available in its own tab. This is a working, newly trained prototype, not a finished camera-quality ISP or a full-data benchmark.

## What changed

The system combines a fixed camera decoder with a small learned editor:

1. LibRaw demosaics the original sensor file and subtracts black. A common camera-code scale, recorded as-shot WB and camera matrix produce **unclipped float32 scene-linear sRGB**. No camera JPEG or embedded thumbnail becomes the training input.
2. A 678,744-parameter CNN analyzes a 256-pixel preview, signed-log RGB and absolute brightness/color statistics. It predicts global exposure, residual color balance, saturation, a 33-knot monotone luminance curve, and local exposure/detail-contrast maps.
3. Adjustments happen before the final display tone/gamut mapping. Values above one and negative matrix values remain available until that intentional display conversion. The unrestricted 3D LUT responsible for the prior tonal-reversal failure is absent from this model.
4. One cached recipe renders previews and native-resolution tiles. The app exports PNG/JPEG or genuine 16-bit sRGB TIFF. Zero grade is a defined neutral RAW development, not the camera's JPEG.

This follows established low-resolution recipe prediction from [HDRNet](https://arxiv.org/abs/1707.02880), constrained interpretable controls from [Exposure](https://arxiv.org/abs/1709.09602), and explicit corruption training motivated by [flare-removal research](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.html). It is an engineering combination, not a novelty claim. See [the consolidated research and next systems](../../docs/raw-direction.md).

## Actual data and protocol

- Original [MIT–Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) DNGs and Expert C 16-bit ProPhoto RGB TIFFs. Targets pass through LittleCMS directly into float sRGB; no 8-bit target intermediate.
- Planned 128 train / 32 validation scenes; **123 train / 32 validation** survived strict checks. Three training files failed RAW level/white-balance checks, one had a fractional crop, and one had inconsistent orientation/geometry. Exclusions and source receipts are preserved in the cloud manifest/preparation report.
- Exact DNG active-area/default-crop tags and orientation are shared by training and app inference. Small target-only float rounding overshoots after resizing are normalized within a 1e-6 tolerance; RAW input values are never clipped for this check.
- All 155 scenes belong to the old outer **training** pool. None of the old 824 outer-test scenes was acquired for this pilot. Validation photos were not used for these RAW model updates. They were used to compare variants and select the app default, so they are validation—not a fresh final test.
- No JPEG checkpoint weights were reused. Consequently, these scores cannot be compared directly with the old JPEG experiment: targets, resolution, baseline and evaluation photos differ. The old JPEG model had already trained on this outer-training pool.
- Four arms trained from scratch for 800 updates each, batch 16, AdamW 0.0002, fixed cosine decay, fixed sampling seed 20260919. Real-data order is matched, with a separate corruption RNG. CNN training uses BF16 on H100; recipes/rendering and final evaluation use FP32. No validation checkpoint stopping. Convergence is not established.
- Shared versus preservation isolates the **augmentation plus consistency training package** on the same architecture. Preservation versus robust adds the veil head. This does not separately isolate augmentation from the consistency loss. Global retains the shared module implementation with local controls disabled; equal total parameter count is not equal active trainable capacity.

## Validation against Expert C

| System | PSNR ↑ | SSIM ↑ | RGB MAE ↓ | ΔE76 ↓ |
|---|---:|---:|---:|---:|
| Neutral RAW development | 16.32 | 0.7938 | 0.1395 | 19.99 |
| Global controls | 21.98 | 0.9006 | 0.0737 | 13.08 |
| Shared global + local (app default) | 23.11 | 0.9075 | 0.0646 | 12.14 |
| Same model + repair/preservation training | 22.91 | 0.9039 | 0.0677 | 12.21 |
| Repair/preservation + extra veil head | 22.53 | 0.9022 | 0.0698 | 12.36 |

Metrics average per photo over valid image content at the 256-pixel pilot resolution. SSIM is a structural similarity index, **not a percentage of correct editing**. ΔE is CIE76, not CIEDE2000. Neutral development is the explicitly defined fixed shoulder baseline, not Lightroom, a camera JPEG or another learned RAW developer.

The shared model reduces average absolute target error by **53.7%** relative to this neutral baseline. Against the learned global arm, its mean PSNR gain is **1.13 dB**; a paired 5,000-resample scene bootstrap gives a 95% interval of **+0.26 to +2.03 dB**. The corresponding SSIM interval crosses zero. These intervals cover scene sampling only, not training randomness. This is one training seed with nondeterministic CUDA operations; repeat seeds and larger held-out data are still needed.

## Selective-repair experiment: an unresolved weakness

Training corruption uses compact feathered ellipses with unclipped linear exposure/veil changes on every fourth batch. Evaluation holds out a diagonal-stripe support family. Outside drift compares a model's grade of the corrupted image against its grade of the clean image, not against the ungraded input.

| System | Faulty-region target MAE ↓ | Outside-region output drift ↓ |
|---|---:|---:|
| Global controls | 0.09365 | 0.00695 |
| Shared global + local (app default) | 0.08761 | 0.01106 |
| Same model + repair/preservation training | 0.08539 | 0.01061 |
| Repair/preservation + extra veil head | 0.09651 | 0.01173 |

The shared local model improves faulty-region target accuracy but changes unaffected regions **more** than the global arm. Preservation training reduces shared-model outside drift by only **4.1%** in this pilot; the paired 95% interval for the change is **−0.00195 to +0.00103**, crossing zero. It does not establish the desired preservation benefit. The extra veil head did not improve mean target fidelity or robustness, so it is **not enabled in the app**. These synthetic faults also do not prove repair of real lens flare or sensor-clipped detail.

## Real RAW execution, export and visual checks

A full file-to-export check on validation scene `fivek:a2339` decoded an actual DNG on a Modal CPU, loaded the trained checkpoint, rendered and inspected a native 2464×1632 uint16 TIFF with its sRGB ICC, and confirmed the original source hash was unchanged. Scene-linear values ranged from −0.357 to 3.794 before display mapping.

Neutral development, automatic grade, and Expert C (`comparison.jpg`)

A separate Mac MPS check used actual Canon ISO800 `fivek:a0007` sensor data entirely in RAM, exact cropping, and **12.17 MP** native output. It substituted only the filesystem-loading boundary with already decoded real bytes; crop/CNN/render/export were real. The independent Modal check above covers the actual path-based file loader.

| Mac measurement | First recorded run |
|---|---:|
| RAW decode | 0.586 s |
| Crop, preview analysis and baseline | 0.240 s |
| Cached preview, median | 25.4 ms |
| Native 16-bit TIFF export | 0.958 s |

A second native-crop review exported in 1.539 s; these are individual measurements, not guaranteed latency. The generated 73 MB TIFFs were verified for bit depth, fine 16-bit codes, dimensions and ICC, then deleted. Original RAWs were not staged on the Mac. Detailed measurements are in the MPS report (`native_mps_check/report.json`).

At 1:1 scale, the grade **amplifies existing luminance/chroma grain and color fringes** on this ISO800 photo. No obvious broad halos, banding or tile seams appeared in the three reviewed crops, but this is not a comprehensive artifact assessment. See native crop review (`native_mps_check/crop_review/report.json`). The pipeline currently has no calibrated RAW denoiser or chromatic-aberration correction, and it does not reconstruct sensor-clipped detail. Fujifilm RAF quality remains unverified without a representative file from the actual camera.

## Runtime, artifacts and next experiments

The completed four-arm H100 function took **108.1 seconds**, including 3.14 seconds loading the verified packed cache, validation, and checkpoint/report saves. Peak allocated GPU memory was about **1.10 GiB**. CPU preparation downloads bounded public source pairs and deletes them after float-cache creation; an 8-worker verification pass creates one tensor pack so the GPU avoids hundreds of small reads.

At [Modal's posted rates](https://modal.com/pricing), this completed function represents about **$0.13 compute** including its requested CPU/RAM. That is not a total bill: dataset preparation, an initial rejected range check, an interrupted optimization attempt, cold starts and verification are additional. The actual invoice was not inspected. The first interrupted global run had slightly different results; this table uses only the final complete comparison.

The full test suite passed **254 tests, with one CUDA-only test skipped locally**; the actual four-model CUDA run separately completed finite forward/backward optimization. CPU/MPS parity, LUT-free monotonicity, exact geometry, RAW headroom, cache tampering, tiled export and TIFF precision have targeted coverage. The running Gradio app was also inspected with RAW selected and the expected controls/export formats visible.

Keep the shared model as the default. The next research priorities are:

1. Validate representative Fujifilm files against a trusted RAW developer, especially backlit/high-ISO scenes. Confirm color, exposure, crop, compression support and noise behavior before camera-specific claims.
2. Add and evaluate calibrated denoising/chroma-aberration handling separately; grading alone cannot suppress the noise it reveals. [SID](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Learning_to_See_CVPR_2018_paper.html) illustrates RAW noise supervision, but its task differs from Expert C grading.
3. Expand the nested RAW development subset to 512 training / 96 validation photos, then more of the frozen training pool. Repeat the leading comparisons with three seeds. Once choices are frozen, a fixed-schedule run can use all 1,924 outer-training scenes and evaluate the untouched 824 once, matching the earlier 70/30 preference.
4. Address outside-region drift with separate augmentation-only and consistency-only controls, genuine unchanged examples, and eventually confidence-weighted person/sky hints or a bilateral/multiscale local renderer. Add a helper only when it beats the same model without it. Do not infer a defect from “this region is bright.”

All four checkpoints, source snapshots, manifest/registry hashes, per-scene metrics and paired comparisons are in this directory. The app uses `shared/model.pt`; the JPEG checkpoint and old benchmark remain unchanged.
