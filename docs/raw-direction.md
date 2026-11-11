# RAW direction

The RAW path keeps scene-linear highlight information through decoding, exposure adjustment, and tone mapping. A shared 679k-parameter CNN predicts global controls, a monotone luminance curve, and local exposure/contrast maps. The [pilot](reports/raw.md) selected the shared model with veil subtraction disabled.

## Pipeline decisions

- LibRaw demosaics and subtracts sensor black; normalization uses a common camera-code scale.
- As-shot white balance and camera-to-sRGB conversion run in float32, retaining negative matrix values and values above one.
- Exact DNG crop tags and orientation are shared by training and app inference.
- A positive-increment curve maps luminance through `Y/(1+Y)` into display range.
- Cached maps serve preview sliders and native tiled export.

This uses preview-to-native rendering ideas from [HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/) and the monotone-curve construction described in [Exposure](https://people.csail.mit.edu/hehaodele/projects/TOG18-exposure/paper.pdf).

The decoder revision fixes channel scaling and accepts unequal green black levels. Canon and Nikon probes retained highlight headroom and closely matched controlled LibRaw references. Details are in the [decoder audit](raw-pipeline-research.md).

## Current boundaries

LibRaw's early black subtraction and demosaicing use integer buffers. Camera conversion uses a daylight matrix approximation; full dual-illuminant DNG profiles and representative Fuji RAF files remain future work. The model's color-balance controls operate in linear sRGB. Exact sensor white-balance edits would require camera-space gains.

Gamut mapping contracts chroma toward equal-luminance gray. Saturated highlights need comparison with a stronger output transform such as [ACES](https://docs.acescentral.com/system-components/output-transforms/). RAW grading currently has no calibrated denoiser or chromatic-aberration stage.

## Next experiments

1. Compare representative Fuji RAWs with a trusted developer, including backlight and high ISO.
2. Expand the nested development set to 512 training / 96 validation scenes, preserving the [data selection](raw-data-research.md#pilot-selection).
3. Repeat shared and preservation variants across three seeds, separating augmentation from consistency loss.
4. Test calibrated denoising separately from grading. [Learning to See in the Dark](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Learning_to_See_CVPR_2018_paper.html) provides paired RAW noise-supervision context.
5. If localization remains weak, compare photometric cues, automatic person/sky masks, and manual masks on the same scenes. [SegFormer](https://arxiv.org/abs/2105.15203) is one candidate helper.

Linear veil augmentation follows [flare-removal research](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.html). Further trials should retain source headroom and score newly clipped regions separately.
