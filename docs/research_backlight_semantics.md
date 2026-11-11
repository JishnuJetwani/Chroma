# Semantic hints for local grading

**Status: proposed experiments.** Start with clipping and texture diagnostics, then compare whether frozen person, clothing, and sky masks improve adjustment placement.

The backlit example contains a white shirt with visible folds, a large bright sky, washed-out vegetation, and a pink flare. Whole-image pure-white coverage is 35.658%; a narrower shirt-interior sample has only 0.182%. Region-specific measurements are useful because the shirt retains detail.

## Candidate helpers

| Helper | Output | Integration notes |
|---|---|---|
| [MediaPipe SelfieMulticlass](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter) | Hair, skin, clothes, accessories, background | Small preview model; its card lists backlit subjects as a failure case |
| [SegFormer-B0 ADE20K](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) | Sky, person, vegetation, other scene classes | 3.75M parameters; NVIDIA research/noncommercial terms |
| [Apple Vision](https://developer.apple.com/videos/play/wwdc2021/10040/) | Person matte | Native Mac framework; pin OS/request revision and quality |
| [Torchvision LR-ASPP](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html) | Person among VOC classes | 3.22M parameters; match Torch/Torchvision versions |

Use soft masks and cache them with the source checksum and helper revision. Fade optional corrections with mask uncertainty. Check the model's accompanying terms before distributing weights. Mac runtime remains to be measured.

## Comparisons

1. **Photometric cues:** clipping, near-white coverage, local gradient energy, luminance percentiles, and minimum-RGB statistics. Keep source-clipping masks separate from headroom after the global LUT.
2. **Frozen masks:** compare the existing recipe, photometric limits, automatic semantic limits, and manual masks on 10–20 development images. Protect bright clothing while allowing detail changes in surviving folds.
3. **Trainable hint adapter:** feed person/sky/clothing probabilities, clipping, detail, and confidence maps into the decoder. Initialize new input-channel weights to zero, freeze most of the original network, and compare at a matched update budget.
4. **Additional operator:** if localization improves but the veil remains, test one bounded affine or additive field separately.

Predict training hints from the actual corrupted inputs. Mask dropout and uncertainty perturbations should teach a fallback when segmentation is unreliable.

## Evaluation

Inspect hair/sky edges, white fabric, equal-luminance color boundaries, narrow gaps, and tile borders. Include people seen from behind, multiple small people, no-person images, high-key scenes, snow, walls, and warm light. Record clipping, fold gradients, chroma drift, halos, and blind preferences.

One recipe and one set of masks should serve the entire native image. Compare tiled/full output and check zero-confidence behavior against the baseline. Benchmark helper loading, first/warm analysis, memory, and cached preview/export separately.

The [dark-channel prior](https://people.csail.mit.edu/kaiming/publications/cvpr09.pdf) motivates some haze statistics but has difficulties with sky and white objects. The [MediaPipe model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf) documents segmentation constraints relevant to this scene. Completed photometric experiments are in the [backlight report](backlight-next-steps.md).
