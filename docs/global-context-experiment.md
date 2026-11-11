# Global context experiment

**Status: configured; FiveK training pending.**

The early global pilot produced nearly constant LUT mixing weights. This experiment adds absolute image statistics to the mixer and increases diversity in the initial LUT bank.

[global_context_pilot.json](../configs/global_context_pilot.json) enables masked RGB/luminance means and standard deviations, identity-centered diverse bases, and initialization strength 0.2. It uses 128 training and 32 validation images, 256-pixel rendering, and ten epochs.

A 20-step CPU conditioning diagnostic improved with diverse initialization; context alone had little effect. The next comparison should vary those two settings independently.

After preparing the [preview manifest](data.md#rebuild-the-development-split), train from scratch:

```bash
uv run --no-sync photograder train --config configs/global_context_pilot.json --device auto
```

Starting from fresh weights preserves the intended LUT initialization. Evaluate with the same scenes, resolution, and metrics as the global baseline.
