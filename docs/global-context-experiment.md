# Optional global context experiment

**Status: configured, not run on FiveK.** `configs/global_context_pilot.json`
adds masked RGB/luminance means and standard deviations to the LUT mixer and
uses diverse, identity-centered LUT bases with initialization strength 0.2.
It uses the same preview manifest, 128 training images, 32 validation images,
256-pixel rendering, and ten-epoch budget as the preview global baseline.

The earlier global pilot showed little image-dependent variation in its mixing
weights. A controlled 20-step CPU diagnostic confirmed that the legacy model
can learn conditional behavior; stronger diverse initialization improved that
toy task, while context alone made little difference. This does not establish
better real-photo grading. See `runs/diagnostics/global_context.json` for the
local diagnostic record when available.

Run from scratch to preserve the requested initialization:

```sh
photograder train --config configs/global_context_pilot.json
```

Do not use `--initialize` for the initial comparison: transferring an existing
LUT bank overrides the diverse initialization. Evaluate the resulting checkpoint
on exactly the same split, image size, controls, and metric version as the
baseline. Isolating context from initialization requires separate ablations;
this combined configuration alone cannot attribute a gain to either change.
