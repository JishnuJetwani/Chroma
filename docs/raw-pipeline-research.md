# RAW pipeline decisions and verification

Research and initial implementation: 2026-10-30. Source review used current upstream rawpy/LibRaw code; the local execution probe used rawpy 0.27.1 and LibRaw 0.22.1. The implementation is `src/photograder/raw_io.py`. No native RAW file was saved locally during verification.

## Decision

Decode to unbalanced, black-subtracted camera RGB, then normalize, white-balance, and transform to D65 linear sRGB in float32. Preserve values below zero and above one through exposure/tone decisions. Apply a deliberate scene-to-display rendering transform before a display-trained model or display export.

This is a conservative initial decoder for RGB Bayer/X-Trans CFA files with usable metadata. It is not a complete floating-point ISP, a colorimetric camera calibration, or a claim of tested Fujifilm support. A representative RAF from the user's exact model and compression mode is still needed.

## What the decoder configuration actually does

```python
raw.postprocess(
    output_color=rawpy.ColorSpace.raw,
    output_bps=16,
    gamma=(1, 1),
    no_auto_bright=True,
    no_auto_scale=True,
    adjust_maximum_thr=0.0,
    bright=1.0,
    use_camera_wb=False,
    use_auto_wb=False,
    user_wb=[1.0, 1.0, 1.0, 1.0],
    highlight_mode=rawpy.HighlightMode.Ignore,
    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
    half_size=False,
    four_color_rgb=False,
    median_filter_passes=0,
)
```

`dcraw_process` copies the mosaic into its processing buffer and subtracts black **before** its conditional `scale_colors` call. Consequently, disabling scaling does not disable black subtraction. It does skip the white-balance/scaling stage. Explicit unity WB is descriptive but redundant while `no_auto_scale=True`. The X-Trans branch uses LibRaw's X-Trans demosaicer; the AHD enum selects quality 3, which selects its three-pass path, rather than applying a Bayer-only AHD algorithm to a 6×6 CFA. [Processing order and demosaic selection](https://github.com/LibRaw/LibRaw/blob/master/src/postprocessing/dcraw_process.cpp).

Black subtraction clips negative results and clears used black corrections in the processing state. The public rawpy black/white metadata must be captured for normalization, but black must **not** be subtracted a second time from `postprocess` output. [Black subtraction source](https://github.com/LibRaw/LibRaw/blob/master/src/preprocessing/subtract_black.cpp), [raw2image inline subtraction](https://github.com/LibRaw/LibRaw/blob/master/src/preprocessing/raw2image.cpp).

With `output_color=raw`, the RGB matrix stage leaves the integer camera values unchanged. Ordinary sRGB/XYZ output applies a matrix and clips its result to uint16. Turning on 16-bit output and a linear gamma therefore does not, by itself, prevent highlight or negative-gamut loss. Exposure correction occurs after scaling; it cannot undo values already clipped during an earlier WB operation. [RGB conversion and exposure implementation](https://github.com/LibRaw/LibRaw/blob/master/src/postprocessing/postprocessing_utils.cpp).

The memory writer ordinarily applies a lookup curve. With gamma `(1,1)`, brightness 1, and auto-bright disabled, it is an identity transform on the 16-bit processing values. It also applies orientation when writing output. The app therefore uses LibRaw orientation exactly once; training can request `apply_orientation=False` and perform its exact reference crop/orientation later. [Memory output source](https://github.com/LibRaw/LibRaw/blob/master/src/postprocessing/mem_image.cpp).

LibRaw documents `no_auto_scale` as a special-purpose switch, particularly intended for custom interpolation. It is not a promise that the complete demosaic pipeline is linear or unclipped. [Parameter documentation](https://github.com/LibRaw/LibRaw/blob/master/doc/API-datastruct.html). X-Trans interpolation itself uses unsigned integer buffers and clamps some interpolated samples. Thus this implementation prevents **WB/CCM display-range clipping**, while below-black noise, sensor saturation, and integer interpolation excursions have separate limitations. [X-Trans implementation](https://github.com/LibRaw/LibRaw/blob/master/src/demosaic/xtrans_demosaic.cpp).

## Normalization and matrix direction

The uint16 dtype is storage precision, not a declaration that sensor white is 65535. Pipeline `raw-camera-float-wb-matrix-v2` uses a **common** normalization scale after LibRaw's per-site black subtraction:

```text
common_black = min(raw.black_level_per_channel)  # all four adjusted channels
scale = raw.white_level - common_black
camera[c] = decoded_camera_integer[c] / scale
balanced[c] = camera[c] * as_shot_wb[c] / as_shot_wb[green]
linear_srgb = M @ balanced
```

All divisions, gains, and the final matrix multiplication happen in float. This follows LibRaw's `scale_colors`: after `maximum -= black`, all channels use the same maximum denominator, with WB as their only different multipliers. Per-channel saturation thresholds describe clipping, not relative sensor sensitivity. Dividing each channel by its own white-minus-black would alter the channel ratios supplied to WB/CCM. [Primary scaling implementation](https://github.com/LibRaw/LibRaw/blob/master/src/postprocessing/postprocessing_utils_dcrdefs.cpp).

Per-channel whites remain available for the sensor saturation diagnostic. Unequal green blacks are supported: LibRaw subtracts each site's black before merging the greens, and the shared scale commutes with that merge. Unequal positive first/second green **WB gains** remain unsupported because those gains cannot be recovered after merging. Missing as-shot WB is rejected rather than silently replaced by daylight or an image-dependent estimate. The source raw mosaic remains untouched.

This corrects the initial v1 implementation's per-channel denominator and overstrict unequal-green-black rejection. Actual FiveK Canon files `a3996` and `a3049` have one-code green black differences estimated by LibRaw even though their DNG BlackLevel tag is zero. `a1342` instead has missing as-shot WB (`[0,1,0,0]`) and remains a legitimate rejection. The old cache and new cache are not interchangeable; the paired renderer version is `fivek-libraw-common-scale-dngcrop-lcms16-float-target-v3`.

`rawpy.rgb_xyz_matrix` exposes LibRaw `cam_xyz`, whereas `rawpy.color_matrix` exposes `cmatrix`. `rawpy` does **not** expose the final active `rgb_cam` through either name. The black property in rawpy 0.27 uses LibRaw's adjusted black-level helper. [rawpy bindings](https://github.com/letmaik/rawpy/blob/main/rawpy/_rawpy.pyx).

`cam_xyz` is an XYZ-to-camera matrix, despite the ambiguous property description. `rgb_cam` converts balanced camera values to linear sRGB. This distinction is explicitly confirmed by the library maintainer. [LibRaw matrix-direction clarification](https://www.libraw.org/node/2099).

For a three-color camera, the implemented conversion reproduces LibRaw's normalized matrix construction:

```python
A = raw.rgb_xyz_matrix[:3].astype(np.float64)  # XYZ -> camera
S = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)  # linear sRGB -> XYZ D65
B = A @ S
B = B / B.sum(axis=1, keepdims=True)
M = np.linalg.inv(B)  # balanced camera -> linear sRGB
linear_rgb = balanced_camera_rgb @ M.T  # HWC arrays
```

The row normalization makes a balanced neutral remain neutral. Inverting `A` directly and applying WB independently does not reproduce this convention. Four-color matrices, absent data, nonfinite entries, invalid white points, and singular/ill-conditioned transforms are rejected. [LibRaw matrix construction](https://github.com/LibRaw/LibRaw/blob/master/src/utils/utils_dcraw.cpp), [LibRaw color constants](https://github.com/LibRaw/LibRaw/blob/master/src/tables/colorconst.cpp).

The chosen matrix is a documented, reproducible **daylight matrix approximation**, not necessarily the matrix selected by LibRaw's ordinary rendering for every file. `cam_xyz` can be absent or differ from the active `rgb_cam` when embedded camera color data are used. There is no automatic fallback to a guessed transform. [Color-data fields](https://github.com/LibRaw/LibRaw/blob/master/doc/API-datastruct.html).

For DNG, do not claim complete dual-illuminant/profile processing: current identification code can select a daylight-related color matrix and combine it with calibration and analog balance. It does not implement the full Adobe look/profile rendering just because a DNG opens. The `cmatrix` selection also depends on camera-matrix/WB configuration. [DNG identification and matrix selection](https://github.com/LibRaw/LibRaw/blob/master/src/metadata/identify.cpp).

## Alternatives and display rendering

An alternative is scaled camera RGB with unity WB (`no_auto_scale=False`, still `output_color=raw`). This gives a convenient 65535 scale and more integer interpolation precision, but interpolation reaches the uint16 ceiling sooner. Another conventional alternative is camera WB with `HighlightMode.Ignore`: LibRaw normalizes WB by its largest gain, rather than the smallest gain used by highlight-clip mode. That provides WB headroom without first boosting channels above their integer range. [Scaling source](https://github.com/LibRaw/LibRaw/blob/master/src/postprocessing/postprocessing_utils_dcrdefs.cpp).

Neither alternative resolves an unreliable final camera matrix. A future LibRaw binding exposing the active `rgb_cam` is preferable to treating `cmatrix` as universally interchangeable. A conventional linear sRGB decode with `highlight_mode=Ignore` and `exp_shift=0.25` can be a clearly labeled fallback preview: it buys two stops before RGB conversion, but still clips negative RGB and offers no universal guarantee against positive clipping. It must not be described as the same unclipped scene data.

A bounded 0–1 LUT cannot infer separate scene highlights once they have been clipped to one. Linear extrapolation beyond a LUT's trained domain retains numbers but does not establish reliable tone behavior. The existing project renderer allows LUT extrapolation for intermediate cascades; that is useful but does not make a display-trained checkpoint a RAW tone mapper.

Recommended architecture: keep the float scene master; predict exposure/tone from a consistently rendered proxy; apply scene exposure/local exposure before a monotonic shoulder and gamut compression; then apply an explicitly display-trained look. Retraining can instead learn a scene-referred operator with a fixed reversible shaper or constrained tone parameters. Preserve negative matrix values until a deliberate gamut operation, and assess bright saturated colors as well as neutral ramps. These are design recommendations, not validated model results.

A complete output transform handles tone, chroma, and gamut together. ACES 2 is a useful reference implementation and comparison baseline; it separates tone mapping from chroma/gamut compression in a perceptual space. A small per-channel Reinhard curve is only a baseline and may alter hues. [ACES output-transform structure](https://docs.acescentral.com/system-components/output-transforms/).

## Camera support and exports

Fujifilm RAW uses `.RAF`; `.AF` is not the normal extension to advertise. The actual decoder must still validate headers/model/compression rather than trust a suffix. X-Trans support exists in LibRaw, while many Fujifilm cameras use Bayer. [RAF format implementation notes](https://libopenraw.freedesktop.org/formats/raf/).

Check the actual runtime version, its supported-camera list, and a real file. The current upstream camera page is version-specific; its full-feature build list is not a guarantee about every packaged build. [LibRaw supported cameras](https://www.libraw.org/supported-cameras), [runtime camera-list API](https://www.libraw.org/docs/API-CXX.html). Rawpy exposes runtime library versions and feature flags; its wheels do not include every optional demosaic pack. [rawpy build information](https://github.com/letmaik/rawpy#optional-features).

16-bit PNG/TIFF can retain more display-output precision than the current 8-bit Pillow path. Quantize once after the final display transform, attach the appropriate sRGB/ICC metadata, and round-trip verify RGB bit depth and values. A 16-bit integer display file is not a substitute for the unclipped float scene master. An optional float TIFF/EXR or NumPy cache can preserve that master with explicit color-space metadata; keep caches bounded because 40 MP float32 RGB requires about 480 MB before overhead. The original RAF/DNG remains the authoritative source.

## Actual verification

- Seventeen synthetic RAW I/O tests passed after the common-scale correction: float WB preserves values above one and negatives; unequal black/white metadata do not change relative signal ratios; the fourth green contributes to common-black selection; black is not subtracted twice; balanced neutrals survive a nontrivial matrix; invalid matrices, WB, levels, and normalization scales are rejected. Together with data tests, 44 checks passed. `ruff` passed on the two decoder/test files.
- A numerical reproduction of the maintainer's published Canon matrix agreed within `3.6e-5` with the older rounded values; the neutral response was `[1,1,1]`.
- An authorized, bounded in-memory fetch of MIT FiveK `a0001-jmac_DSC1459.dng` read exactly 5,248,022 bytes. Decode succeeded with rawpy 0.27.1 / LibRaw 0.22.1. No native RAW was written to disk.
- With orientation disabled, output was float32 `(2014, 3040, 3)`, range `[-0.11663587, 2.35297823]`, sensor saturation fraction `0.0`. Black levels were zero, whites were 4095, WB was approximately `[2.12890506, 1, 1.54687417]`. This demonstrates retained positive headroom and negative RGB on an actual file.
- RAW and visible geometry were 3040×2014 with zero margins and flip 0. Rawpy's derived crop `(16,8,3007,1999)` differs from the exact DNG default crop tags `(15,7,3008,2000)`. The low-level decoder preserves the full visible field. Paired training and native app DNG inference now use `crop_dng_raw` to apply the exact DNG crop, subtract the visible origin, and orient once with matching geometry.
- Orientation was tested by changing TIFF Orientation to 6 **only in RAM**. The oriented decode was `(3040,2014,3)` and exactly matched a clockwise 90-degree rotation of the unoriented output (maximum absolute error zero).
- Small probe artifacts are in `runs/raw_20261030/decode_probe`: an unclipped float preview NPZ, metadata JSON, and a diagnostic shoulder-rendered PNG. The PNG uses stride sampling and is not a production-quality resized/exported image.
- An exact-cropped, antialiased a0001 comparison shows the initial float-WB decode and a LibRaw-WB reference agree closely: display MAE `0.0007383`. Expert C has substantially stronger contrast/saturation; no gross channel swap or WB error was visible. The equal-level a0001 normalization is unchanged between v1 and v2. See `a0001-audit-comparison.png` and JSON.
- The v2 unequal-green-black test used a bounded in-memory 7,602,968-byte fetch of Canon `a3996`. Its common scale is `4095−127=3968`; decoded range is `[-0.147739,1.858784]`, with 215 component values above one and 128,088 below zero retained. An independent LibRaw `scale_colors` camera-RGB reference used WB normalized to its largest gain, then undid that scale and applied the same float CCM. Its uint16 maximum was 31,723 with **zero** 65,535 ceiling samples. Antialiased display MAE was `0.00090584`; the maximum `0.03936` reflects localized integer/demosaic differences. Visual inspection found matching overall tone and color. See `a3996-common-scale-comparison.png` and JSON. This validates the common-scale path on one actual Canon file, not every camera.
- Still required: the user's representative Fujifilm RAF, fine-detail/hot-highlight visual review, and color/exposure comparison with a trusted RAW rendering. This single Nikon-origin DNG does not validate Fuji decoding quality or all supported formats.
