# RAW decoder

[raw_io.py](../src/photograder/raw_io.py) decodes black-subtracted camera RGB, then applies normalization, as-shot white balance, and a camera-to-sRGB matrix in float32. The implementation was checked with rawpy 0.27.1 and LibRaw 0.22.1.

## LibRaw configuration

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

LibRaw subtracts black before the optional scaling stage. The returned camera values therefore already have black removed. `no_auto_scale=True` leaves white balance and scaling to the float pipeline. With this configuration, X-Trans uses LibRaw's quality-3 X-Trans demosaicer. [Processing order](https://github.com/LibRaw/LibRaw/blob/master/src/postprocessing/dcraw_process.cpp).

Black subtraction and demosaicing still use integer buffers and can clip below-black values or interpolation excursions. The float stages preserve later white-balance headroom and negative matrix outputs. Gamma `(1,1)`, brightness one, and disabled auto-bright leave the writer's lookup curve at identity. Orientation is applied once, after the exact crop where required. [Black subtraction](https://github.com/LibRaw/LibRaw/blob/master/src/preprocessing/subtract_black.cpp) · [Memory output](https://github.com/LibRaw/LibRaw/blob/master/src/postprocessing/mem_image.cpp).

## Common normalization

Decoder version `raw-camera-float-wb-matrix-v2` uses:

```text
common_black = min(raw.black_level_per_channel)
scale = raw.white_level - common_black
camera[c] = decoded_camera_integer[c] / scale
balanced[c] = camera[c] * as_shot_wb[c] / as_shot_wb[green]
linear_srgb = M @ balanced
```

A common scale preserves channel ratios and follows [LibRaw's scaling convention](https://github.com/LibRaw/LibRaw/blob/master/src/postprocessing/postprocessing_utils_dcrdefs.cpp). Per-channel white levels are retained for saturation diagnostics. Unequal green black levels are supported; unequal positive green WB gains and missing as-shot WB are rejected.

This fixes the initial per-channel denominator. The paired cache version is `fivek-libraw-common-scale-dngcrop-lcms16-float-target-v3`; caches from the earlier normalization need regeneration.

## Camera matrix

`rawpy.rgb_xyz_matrix` exposes LibRaw's `cam_xyz`, an XYZ-to-camera matrix. The implementation constructs a balanced-camera-to-linear-sRGB matrix by row-normalizing its product with the sRGB-to-XYZ matrix, then inverting it:

```python
A = raw.rgb_xyz_matrix[:3].astype(np.float64)
S = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
B = A @ S
B = B / B.sum(axis=1, keepdims=True)
M = np.linalg.inv(B)
linear_rgb = balanced_camera_rgb @ M.T
```

Row normalization preserves balanced neutrals. Invalid, singular, nonfinite, or four-color matrices are rejected. This is a daylight matrix approximation; full dual-illuminant DNG profile rendering remains future work. Rawpy's public matrices differ from LibRaw's final active `rgb_cam`. [Matrix-direction clarification](https://www.libraw.org/node/2099) · [LibRaw construction](https://github.com/LibRaw/LibRaw/blob/master/src/utils/utils_dcraw.cpp) · [rawpy bindings](https://github.com/letmaik/rawpy/blob/main/rawpy/_rawpy.pyx).

## Geometry and verification

DNG training and app inference share `crop_dng_raw`, which applies `ActiveArea` and exact default-crop tags before orientation. For `a0001`, the metadata crop starts at `(15,7)` with size `3008×2000`. See [source geometry](raw-data-research.md#geometry-and-color).

| Real-file probe | Retained linear RGB range | Display MAE against controlled LibRaw reference |
|---|---|---:|
| Nikon a0001 | −0.116636 to 2.352978 | 0.0007383 |
| Canon a3996 | −0.147739 to 1.858784 | 0.00090584 |

The Canon probe exercises unequal green black levels with common scale `4095−127=3968`. The reference had zero uint16 ceiling samples. A rotated a0001 decode exactly matched the corresponding 90-degree rotation. Numerical tests cover normalization, neutral response, invalid metadata, headroom, and source preservation.

## Camera support and export

Support depends on the installed [LibRaw camera list](https://www.libraw.org/supported-cameras), metadata, and compression. Representative Fujifilm RAF validation is still pending. The loader rejects files above its 60-megapixel guard.

The native float source remains available until tone and gamut mapping. Final PNG/JPEG output is 8-bit sRGB; TIFF output is 16-bit sRGB with ICC metadata. The [RAW pilot report](reports/raw.md) records native exports and noise observations.
