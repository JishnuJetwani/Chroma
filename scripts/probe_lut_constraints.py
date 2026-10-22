"""Fixed-budget shape-only LUT probe; no images, targets, or test split are used."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from photograder.lut_safety import diagnose_neutral_ramp
from photograder.runtime import file_sha256, load_checkpoint, save_checkpoint
from photograder.training_constraints import lut_ramp_losses, lut_shape_losses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/full-7030-20261012/joint/model.pt")
    p.add_argument("--output", default="runs/backlight_20261021/lut_shape_probe")
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    out = Path(args.output)
    if (out / "model.pt").exists():
        raise FileExistsError(out / "model.pt")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260919)
    model, checkpoint = load_checkpoint(args.checkpoint, "cpu")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tables = model.bank.tables
    tables.requires_grad_(True)
    original = tables.detach().clone()
    opt = torch.optim.Adam([tables], lr=2e-4)
    config = {
        "description": __doc__,
        "steps": args.steps,
        "learning_rate": 2e-4,
        "fidelity": 1.0,
        "monotonicity": 0.01,
        "curvature": 1e-5,
        "ramp_monotonicity": 0.01,
        "ramp_samples": 257,
        "source_checkpoint": args.checkpoint,
        "source_sha256": file_sha256(args.checkpoint),
        "data_used": None,
        "validation_used": None,
        "seed": 20260919,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    with torch.no_grad():
        before = diagnose_neutral_ramp(tables)
    history = []
    started = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        shape = lut_shape_losses(tables)
        ramps = lut_ramp_losses(tables, samples=257)
        fidelity = (tables - original).square().mean()
        loss = (
            fidelity
            + 0.01 * shape["lut_monotonicity"]
            + 1e-5 * shape["lut_curvature"]
            + 0.01 * ramps["ramp_monotonicity"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite constraint objective")
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == args.steps - 1:
            row = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "fidelity": float(fidelity.detach()),
                **{k: float(v.detach()) for k, v in shape.items()},
                "ramp_monotonicity": float(ramps["ramp_monotonicity"].detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    elapsed = time.perf_counter() - started
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.update(
        {
            "trained": True,
            "pilot": True,
            "experiment": "shape-only LUT postfit; no new supervised image training",
            "source_checkpoint_sha256": config["source_sha256"],
        }
    )
    save_checkpoint(out / "model.pt", model, metadata=metadata)
    with torch.no_grad():
        after = diagnose_neutral_ramp(tables)
    report = {
        "config": config,
        "seconds": elapsed,
        "before": before,
        "after": after,
        "table_delta_mae": float((tables - original).abs().mean().detach()),
        "table_delta_max": float((tables - original).abs().max().detach()),
        "history": history,
        "checkpoint_sha256": file_sha256(out / "model.pt"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"seconds": elapsed, "checkpoint": str(out / "model.pt")}))


if __name__ == "__main__":
    main()
