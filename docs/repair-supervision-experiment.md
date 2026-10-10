# Optional repair supervision and combined-fault curriculum

**Status: implemented and unit-tested, not trained as a photo experiment.**
`configs/repair_supervised_pilot.json` enables two independently configurable
additions to the repair stage. Existing configurations keep both disabled.

- `training.combined_faults_after_micro_step` defaults to `null`. An integer
  enables combined faults starting at that zero-based training microbatch index,
  before gradient accumulation. The optional configuration uses 128: with 128
  images and batch size two, this introduces combinations after two epochs.
  The existing corruption generator then samples two operations for 35% of
  non-identity examples. The rule also applies to synthetic batches in joint
  training if explicitly configured there.
- `loss.exposure_aux` defaults to zero. A positive value adds a validity-masked
  L1 loss in exposure stops **only during repair training**, using the generator's
  known inverse for pure exposure faults and the zero inverse for identity
  examples. Contrast and combined faults have unknown inverse labels and
  contribute zero. Human-edited pairs never provide inferred inverse labels.

The exposure loss supervises the full-resolution exposure field after the
renderer's actual guided/bilinear upsampling and bounds, not a differently
resized target map. This preserves operator consistency, but a low-resolution
smooth model cannot represent every synthetic inverse exactly. The optional
weight is 0.05. This does not add a contrast inverse label or a new penalty
forcing contrast parameters toward identity.

The auxiliary applies to recipes with one exposure map (including the shared
local model); cascade and RGB/grid-only recipes receive zero. It covers valid
pixels outside the fault as well, where known inverse exposure is zero, and
excludes padding/privacy pixels. The global grading branch remains bypassed.

Validation remains a fixed single-fault/identity protocol with the same seed
rule and families throughout training, regardless of the training curriculum.
Its settings are recorded in checkpoint metadata. Changing the curriculum or
auxiliary loss requires a new experiment, rather than changing a resumed run.

```sh
photograder train --config configs/repair_supervised_pilot.json \
  --initialize runs/global_pilot/best.pt
```

This combined configuration cannot isolate the effect of supervision from the
effect of combined faults. Run separate settings with just one enabled, using
the same starting global checkpoint, data, update budget, and evaluation protocol.
No improvement in real-photo correction or research novelty is claimed.
