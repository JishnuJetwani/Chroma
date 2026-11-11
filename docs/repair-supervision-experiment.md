# Repair supervision experiment

**Status: implemented and tested; photo-training run pending.**

[repair_supervised_pilot.json](../configs/repair_supervised_pilot.json) adds a combined-fault curriculum and direct exposure supervision to the repair stage.

| Setting | Default | Experiment |
|---|---|---|
| `training.combined_faults_after_micro_step` | `null` | `128`: begin combinations after two epochs at batch two over 128 images |
| `loss.exposure_aux` | `0` | `0.05`: supervise the known inverse exposure field |

After the curriculum threshold, 35% of non-identity examples receive two operations. The exposure loss applies to pure exposure faults and unchanged examples, whose inverse labels are known. Contrast and combined faults contribute zero to this auxiliary term.

Supervision uses the rendered exposure field after upsampling and bounds. Valid pixels outside the fault have a zero exposure target; padding and privacy masks are excluded. The global branch remains bypassed during repair. This auxiliary supports recipes with a single exposure map.

Prepare the [preview manifest](data.md#rebuild-the-development-split), train the preview global baseline, then start the repair experiment:

```bash
uv run --no-sync photograder train --config configs/preview/global.json --device auto
uv run --no-sync photograder train --config configs/repair_supervised_pilot.json \
  --initialize runs/preview_reproduction/global/best.pt --device auto
```

Validation keeps the same single-fault/identity protocol throughout training. Compare each addition separately using the same initialization, data, and update budget before testing both together. A changed curriculum or loss starts a new run.
