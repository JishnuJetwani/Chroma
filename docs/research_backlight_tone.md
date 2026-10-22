# Backlight research: photographic tone and exposure-aware rendering

Independent direction-1 memo, 2026-10-21. Read-only inspection of code and the supplied 640×360 JPEG; no rendering changes, model runs, image uploads, or large downloads. Experiments below are proposals, not measured improvements.

## What this image and renderer permit

The photograph contains a bright sky/sun, broad washed-out foreground, surviving white-shirt folds, and a distinct pink flare ghost. Measured directly from the decoded JPEG:

| Region | All RGB = 255 | All RGB ≥ 250 |
|---|---:|---:|
| Whole image | 35.66% | 45.57% |
| Top half, y=0:180 | 55.95% | 69.51% |
| Approximate shirt ROI, x=316:446, y=163:344 | 1.95% | 4.88% |
| Bottom quarter, y=270:360 | 0% | 0% |

These counts establish absent encoded detail, not the camera's original sensor-clipping history. Darkening a uniform white sky produces uniform gray; recovering clouds there requires additional evidence or invented content. Shirt/foreground contrast can improve using the surviving signal. Wu et al. similarly separate saturated light sources from recoverable flare-corrupted content. [Flare removal paper, §5.1](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.pdf)

[The current local operator](../src/photograder/renderer.py) computes `l=log2(Y)`, guided base `b`, detail `d=l-b`, then `l_out=b+2^c*d+e`. It already supports edge-aware decomposition, ±1-stop exposure, and 0.5–2× detail contrast. At this image size its guided radius is 13 pixels; coefficient averaging doubles the support. The exposure map can *learn* broad tonal changes, but the contrast control itself changes only the residual detail. A nearly constant washed-out region barely responds to it.

The renderer preserves intermediate over-range values through cascades, which is desirable, but [the terminal operation](../src/photograder/renderer.py) is independent RGB clipping. The RGB ratio operation preserves linear chromaticity before clipping; it does not guarantee perceived hue/saturation or display-gamut membership. Strong tone changes can require separate color correction. [Mantiuk et al., §3](https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk09cctm.pdf)

The JPEG is display-referred. Inverting its sRGB transfer function does not recover original scene radiance or undo an unknown camera tone curve. All physical corrections below are therefore approximate.

## Experiment 1 — Protect surviving highlights at final output

**Hypothesis:** A terminal shoulder and chroma reduction preserve more shirt texture and avoid colored highlight clipping than the present clamp. This prevents *new* damage; it cannot restore the white sky.

Freeze the checkpoint and compare the current renderer with an opt-in output operator applied once after every LUT/local/cascade stage. For nonnegative rendered linear luminance `z`, try:

```text
T_k(z) = z                                         if z ≤ k
       = k + (1-k)·(z-k)/((1-k)+(z-k))              otherwise
k ∈ {0.65, 0.80}
```

This proposed shifted rational shoulder is continuous with unit derivative at its knee and approaches 1. Its motivation is photographic highlight compression, not a verbatim implementation of Reinhard's operator. [Reinhard et al., §3.1](https://www-old.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf)

Recombine `C = RGB·T_k(Y)/max(Y,1e-6)`. For a cheap, explicit gamut baseline, let `q=T_k(Y)`, `d=C-q` per channel, and choose:

```text
t = min(1, min[(1-q)/d_c where d_c>0], min[q/(-d_c) where d_c<0])
RGB_out = q + t·d
```

This contracts toward a neutral of equal luminance and is in gamut for `0≤q≤1`; guard empty minima and near-zero denominators. It preserves luminance and RGB opponent direction, **not constant perceptual hue**. Compare the same shoulder with max-RGB normalization as a ratio-preserving baseline. If color patches visibly shift, a later perceptual-space mapping should hold hue while reducing chroma; ACES demonstrates that architecture, but adopting its whole scene-referred transform on this JPEG would be a much larger intervention. [ACES gamut compression](https://docs.acescentral.com/system-components/output-transforms/technical-details/gamut-compression/)

Because a shoulder can turn existing sky-white into gray, separately test restoring the input in the large near-white component connected to the upper image edge: start with all-channel 250–255 smooth weighting and feather 4–8 pixels. Keep this mask separate from recoverable shirt highlights. A luminance threshold alone cannot distinguish a sun from a flare ghost. This is a white-preservation policy, not reconstruction. darktable likewise distinguishes tone mapping from repairing already damaged signal. [filmic RGB documentation](https://docs.darktable.org/usermanual/development/de/module-reference/processing-modules/filmic-rgb/)

**Bound:** two knees × two chroma treatments, plus the original baseline; no training. Record newly clipped channels outside source-white areas, shirt-fold gradients, skin/green hue drift, and any sky-mask seam. Preserve exact bypass and old recipes through a versioned opt-in. Keep `pre_gamut` diagnostics before this operator so clipping prevention cannot conceal a poor upstream recipe.

**Failure cases:** gray whites without the mask; excessive highlight desaturation; a visible mask transition; shoulder compression erasing low-contrast folds; modified JPEG colors that do not behave as scene-linear data.

## Experiment 2 — Give broad tones their own adjustment

**Hypothesis:** Explicit base contrast can remove the flat appearance more effectively than increasing fine-detail contrast. Durand–Dorsey's key separation is broad/base tone versus detail; its HDR example compresses the base, whereas this washed-out JPEG may need modest *expansion*. [Original project and paper](https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/)

Reuse the guided filter and test:

```text
b = GuidedFilter(log2(Y), radius=r, eps=0.05)
d = log2(Y)-b
l_out = b + e + w·(k_b-1)·(b-p) + (k_d-1)·d
p = log2(0.35)
k_b ∈ {1.15, 1.30}; k_d=1.10; r ∈ {13, 32} at width 640
```

Use a fixed feathered foreground/subject support `w` for the first representational test; taper it to zero in the clipped sky and exclude the pink ghost. Keep the checkpoint's exposure `e` unchanged. With `k_b>1`, values below the pivot darken and values above it brighten; Experiment 1 controls the newly lifted highlights. Four combinations suffice. Guided filtering is a compatible efficient primitive, although “edge preserving” does not excuse checking hair/silhouette halos. [He et al., §§3–4](https://mmlab.ie.cuhk.edu.hk/2010/eccv10_Guided.pdf)

If the fixed-parameter ceiling is useful across 12–20 held-out backlit/white-object images, add one bounded base-contrast recipe field, initially identity. Supply log-luminance percentiles and source-clipped fraction as absolute cues to its prediction; the current optional global context contains mean/std, but not these cues. Increase tile halos to twice the largest guided radius, and calculate any global statistics once per image.

**Failure cases:** expanded JPEG noise, black halos against sky, “crunchy” cloth, global brightness changes from a poor pivot, and flattening an intentionally soft backlit look. Subject whites and white clouds need negative/no exposure in different circumstances; sign-constrained highlight/shadow heads alone do not distinguish them.

## Experiment 3 — Test additive veil separately from tonal gain

**Hypothesis:** A conservative positive RGB veil field can correct the foreground's raised dark floor and muted chroma, which scalar exposure cannot directly subtract. Flare synthesis commonly adds a light field in approximately linear space. [Wu et al., Eq. 6](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.pdf)

Before the LUT, test `J=max(I-α·w·V,0)` in decoded linear RGB, where `V=v·(1,1,1)`, `v∈{0.01,0.025,0.04}`, and `α=1`. Use the same fixed feathered foreground support, zero inside clipped sky/sun and ghost core. Compare each to an exposure-only correction matched in mean foreground luminance; this exposes whether subtraction restores contrast/chroma rather than merely darkening. Three candidates plus matched controls are sufficient. A later learnable version could predict a smooth positive 4×4 or 8×8 veil with magnitude/curvature penalties and clean-image zero-veil supervision. Existing `rgb_residual` and bilateral affine paths operate in encoded sRGB, so they are not this linear-light correction.

Do not silently replace this with atmospheric dehazing. Atmospheric formation is `I=tJ+(1-t)A`, with attenuation as well as airlight; lens glare has different geometry and can be additive. The dark-channel prior is based on dark pixels in local natural-image patches and excludes sky in its premise. A bright white shirt supplies weak evidence for that premise, so blindly estimating haze from its high dark channel can darken it or create color casts. [He et al., original haze-removal project](https://people.csail.mit.edu/kaiming/cvpr09/index.html)

**Failure cases:** removed legitimate ambient light, crushed foreground channels, biased veil color, or a smooth field leaving the structured pink ghost intact. Restoring the ghost-covered shirt where information is absent needs a separate restoration/inpainting direction. Evaluate un-clipped shirt/foreground ROIs, neutral-shirt chroma, negative pre-clamp residuals, and boundary bands; do not score the amount of sky darkening as recovery.

**Priority:** first prove the rendering ceiling with fixed controls, then train only the controls that help. All trials should include clean/high-key/white-object negative controls and visual preference/artifact judgments; lower paired loss alone is insufficient for this photograph.
