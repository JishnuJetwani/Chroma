# Actual RAW data acquisition and geometry

Checked 2026-10-30. **Official FiveK original DNGs and Expert C TIFFs are individually downloadable.** This gives a practical RAW training path without acquiring the complete archive. The source preparation module is [raw_acquisition.py](../src/photograder/raw_acquisition.py). S24 remains a useful later source, but a bounded downloadable S24 pair has not been verified.

## Verified FiveK source access

The [official MIT–Adobe FiveK page](https://data.csail.mit.edu/graphics/fivek/) links 5,000 original DNGs and five 16-bit, losslessly compressed ProPhoto RGB TIFF renditions for each scene. We fetched its 4,629,128-byte HTML index and parsed the exact DNG and Expert C links, requiring one correctly matched pair per scene. The retained index SHA256 is `cc11c048a6c29459628cd0989ba28324857988eac10894773b8a20dbb9fd3127`.

Three pairs were checked using HEAD and exact HTTP 206 ranges. The ranges contained valid TIFF/DNG headers, not an HTML download page. These examples are all in the existing outer training split.

| Scene | Camera | Original DNG bytes | Expert C TIFF bytes | Oriented target dimensions |
|---|---|---:|---:|---|
| a0001 | Nikon D70 | 5,248,022 | 31,609,888 | 3008 × 2000 |
| a0005 | Canon EOS 30D | 5,178,006 | 38,897,810 | 2336 × 3504 |
| a0007 | Canon EOS Digital Rebel XSi | 14,953,636 | 63,344,800 | 2848 × 4272 |

The smallest inspected pair is [a0001 DNG](https://data.csail.mit.edu/graphics/fivek/img/dng/a0001-jmac_DSC1459.dng) and [a0001 Expert C TIFF](https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/a0001-jmac_DSC1459.tif). Header inspection confirmed CFA sensor pixels in a DNG SubIFD and a 16/16/16-bit RGB target with a 940-byte embedded ProPhoto RGB ICC profile. The root DNG IFD is a small 8-bit preview; it must not be treated as the RAW image.

No original photo was written to local storage by this acquisition work. Small header ranges were inspected in memory; index, access evidence and the source plan are retained under `data/raw_fivek_plan`. This preserves the existing 20 GiB local free-space rule. The original acquisition module's storage guards were not changed.

## Frozen pilot and transfer estimate

The source plan reuses [the frozen 70/30 registry](../configs/splits/cloud_full_7030.json). It selects all development data from its 1,924 outer-training scenes and excludes all 824 outer-test scenes. The 192 smallest `SHA256("raw-fivek-inner-val-v1:" + scene_id)` values form a fixed inner-validation partition; the other 1,732 scenes form inner training. Within each partition, `SHA256("raw-fivek-pilot-v1:" + scene_id)` determines the nested acquisition order.

The first plan contains **128 training and 32 validation scenes**. Increasing the request to 512/96 preserves all earlier memberships and never moves a scene between partitions. Different camera renderings and alternate expert targets must continue to use the same `fivek:aNNNN` scene identity and partition.

HEAD requests verified all 320 source files in the first plan. Their exact sum is **10,210,570,990 bytes**, approximately **9.51 GiB** of source transfer. The largest pair totals **142,470,550 bytes**, approximately **135.9 MiB**. This is a transfer estimate, not a requirement to retain 9.51 GiB: a cloud CPU worker can download one scene, create a bounded training cache, persist checksums and preparation metadata, and then remove that scene's DNG/TIFF staging files before fetching the next pair. No cloud job was launched by this acquisition work.

Create or inspect a source plan without fetching photo bodies:

```sh
python -m photograder.raw_acquisition \
  --registry configs/splits/cloud_full_7030.json \
  --output data/raw_fivek_plan \
  --train-count 128 --val-count 32 --probe-sizes
```

`download_pair(record, stage, max_bytes, min_free_bytes=20*1024**3)` is the cloud preparation entry point. `stage` is a per-scene directory. Records contain the exact `input_url`, `target_url`, `input_bytes`, `target_bytes`, scene identity and split. Downloads reject unexpected response status, changed sizes or modification timestamps, budget overruns, truncated responses and non-TIFF signatures. Each file receives a SHA256 receipt. Resume requires matching URLs, lengths and digests; an existing file without a valid receipt is rejected. Per-file hashes are first measured during acquisition; they are not claimed to be hashes published by the authors.

The preparation layer should retain source hashes, exact crop and orientation, decoder/color transform versions, output array shapes, cache SHA256 and a rendering version. Its proposed cache uses float32 common linear sRGB inputs and float32 display sRGB targets, plus a valid mask. Original sensor DNGs in FiveK are not denoised ground truth. The [Modular Neural ISP dataset notes](https://raw.githubusercontent.com/SamsungLabs/modular_neural_isp/main/datasets/README.md) explicitly describe creating separate Lightroom-denoised DNGs for their experiments; that is additional processing, not an original FiveK asset provided here.

## Crop and orientation are part of the pair

Matching scene names does not make the raw sensor rectangle identical to the retouched image. The three inspected DNGs contain these actual metadata values:

| Scene | Full sensor W × H | ActiveArea `[top,left,bottom,right]` | DefaultCropOrigin `[x,y]` | DefaultCropSize `[w,h]` | EXIF orientation |
|---|---|---|---|---|---:|
| a0001 | 3040 × 2014 | `[0,0,2014,3040]` | `[15,7]` | `[3008,2000]` | 1 |
| a0005 | 3596 × 2360 | `[12,74,2360,3596]` | `[10,7]` | `[3504,2336]` | 8 |
| a0007 | 4312 × 2876 | `[18,22,2874,4312]` | `[10,5]` | `[4272,2848]` | 6 |

All three default crop sizes exactly match the corresponding TIFF dimensions after orientation. In a0001, a naive centered crop would start at x=16; the DNG specifies x=15. A one-pixel offset is avoidable and should not be hidden by resizing the whole sensor image to the target.

`dng_crop_metadata(path_or_buffer)` uses tifffile to read metadata without decoding pixels. It follows SubIFDs, selects the largest CFA sensor image, handles rational crop tags using their denominators, and rejects invalid rectangles or noninteger crops instead of silently rounding. It returns ActiveArea, crop origin/size, default scale, orientation, sensor size and the chosen IFD offset/path. The caller must account for any margins already removed by RawPy, apply the crop before EXIF orientation, then verify the paired dimensions. A nonunit DefaultScale also requires explicit handling. A geometry fallback is an approximation that must be recorded and visually checked.

The [authors' FiveK preparation script](https://raw.githubusercontent.com/SamsungLabs/modular_neural_isp/main/datasets/adobe_5k_preprocessing.py) also treats orientation and sensor/target geometry separately; it undoes target orientation and center-crops the raw image. Our inspected DNG crop tags provide more precise placement for these pairs. Geometry agreement still does not establish subpixel registration after all rendering operations; representative pairs need visual inspection before interpreting reconstruction metrics.

The source module passed checks for nested 128/32 → 512/96 selections, exclusion of every outer-test scene, real DNG SubIFD metadata for all three examples, fractional-crop rejection, the preflight network cap, SHA receipt creation, resume and corruption rejection. Ruff passed. These source checks do not establish a trained model's image quality.

## S24 access and representation

The [official S24 README](https://raw.githubusercontent.com/SamsungLabs/time-aware-awb/main/s24-raw-srgb-dataset/README.md) describes 3,224 scenes with demosaiced, black-normalized three-channel RAW PNGs, denoised RAW PNGs, Lightroom style targets and capture metadata. Its camera JPEGs have a documented pixel shift; use the Lightroom style targets. Privacy masks identify excluded blurred areas, while secondary-illuminant masks have a different meaning.

Its current download links still lead to Sync shares. The earlier saved access probe verified fifteen training split-archive parts, beginning with a displayed 20.0 GB part. This investigation did not establish individual-photo extraction, exact multipart sizes or a small paired download. The official GitHub repository tree contains figure images and model configuration files, but no downloadable real RAW/target sample. No S24 source file was acquired, and S24 should not be described as part of this pilot. The existing access probe (generated locally: `../data/s24_access_probe.json`) remains the evidence for that gate.

The existing `import_s24` adapter reads the genuine PNG-16 source, uses camera-estimated white balance and CCM, then clamps, applies the sRGB transfer and saves an 8-bit input PNG. That adapter is a display-rendered compatibility path. Future RAW training should cache the floating linear color conversion before its display transform and clipping, preserve official train/validation/test membership, exclude privacy masks and verify orientation on actual pairs. The inspected README distinguishes camera and ground-truth illuminants; ground-truth white balance must not be used as an inference-time input feature.

## Source terms

FiveK has two research licenses and per-file applicability lists. Both restrict the grant to research without commercial advantage, require notices with copies and prohibit implied endorsement. The original [Adobe license](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobe.txt), [Adobe/MIT license](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobeMIT.txt) and file lists are retained beside the source plan. These are dataset terms, not a claim of commercial permission for a downstream product.

The S24 repository's accompanying [LICENSE.md](https://raw.githubusercontent.com/SamsungLabs/time-aware-awb/main/LICENSE.md) contains CC BY-NC-SA 4.0 terms. Some author script headers use a different abbreviated CC BY-NC label; preserve the actual accompanying license and dataset attribution rather than removing either. No source dataset is redistributed in this repository.
