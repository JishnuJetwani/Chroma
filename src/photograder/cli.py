"""Command-line entry points; heavyweight modules are imported only when needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _config(path):
    return json.loads(Path(path).read_text())


def _overrides(config, args):
    config.setdefault("training", {})
    config.setdefault("data", {})
    for name in ("epochs", "max_steps", "device", "seed", "batch_size"):
        if getattr(args, name, None) is not None:
            config["training"][name] = getattr(args, name)
    for name in ("manifest", "limit", "image_size", "val_limit"):
        if getattr(args, name, None) is not None:
            config["data"][name] = getattr(args, name)
    if getattr(args, "output", None):
        config["output_dir"] = args.output
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="photograder", description="Train, evaluate, and use Selective Grade locally."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check device, environment and model size")
    acquire = sub.add_parser("acquire", help="Fetch only paired FiveK JPEGs using HTTP ranges")
    acquire.add_argument("--output", default="data/fivek_pilot")
    acquire.add_argument("--full", action="store_true")
    acquire.add_argument("--train-count", type=int, default=16)
    acquire.add_argument("--val-count", type=int, default=4)
    acquire.add_argument("--test-count", type=int, default=4)
    acquire.add_argument("--workers", type=int, default=4)
    verify = sub.add_parser("verify-data")
    verify.add_argument("manifest")
    performance = sub.add_parser(
        "benchmark-inference", help="Measure cached preview and native-resolution PNG export"
    )
    performance.add_argument("image")
    performance.add_argument("--checkpoint", required=True)
    performance.add_argument("--output", required=True)
    performance.add_argument("--device", default="auto")
    performance.add_argument("--repeats", type=int, default=3)
    for name in ("train", "benchmark"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
        item.add_argument("--manifest")
        item.add_argument("--output")
        item.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
        item.add_argument("--limit", type=int)
        item.add_argument("--val-limit", type=int)
        item.add_argument("--image-size", type=int)
        item.add_argument("--batch-size", type=int)
        item.add_argument("--seed", type=int)
        if name == "train":
            item.add_argument("--epochs", type=int)
            item.add_argument("--max-steps", type=int)
            group = item.add_mutually_exclusive_group()
            group.add_argument("--resume")
            group.add_argument("--initialize")
        else:
            item.add_argument("--steps", type=int, default=200)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--image-size", type=int, default=512)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--save-images", action="store_true")
    evaluate.add_argument(
        "--controls", action="store_true", help="Include held-out corruption locality tests"
    )
    evaluate.add_argument("--seed", type=int, default=42)
    grade = sub.add_parser("grade")
    grade.add_argument("image")
    grade.add_argument("--checkpoint", required=True)
    grade.add_argument("--output", required=True)
    grade.add_argument("--style")
    grade.add_argument("--grade-strength", type=float, default=1)
    grade.add_argument("--local-strength", type=float, default=1)
    grade.add_argument(
        "--treatment",
        choices=("model", "backlight"),
        default="model",
        help="Optional experimental treatment for bright, veiled photos",
    )
    grade.add_argument("--device", default="auto")
    grade.add_argument("--recipe", help="Also save portable recipe tensors")
    app = sub.add_parser("app")
    app.add_argument(
        "--checkpoint", help="JPEG or RAW checkpoint; discovers local trained models when omitted"
    )
    app.add_argument("--raw-checkpoint", help="Optional RAW checkpoint alongside a JPEG checkpoint")
    app.add_argument("--device", default="auto")
    app.add_argument("--port", type=int, default=7860)
    app.add_argument("--export-dir", default="exports")
    matrix = sub.add_parser("experiments")
    matrix.add_argument("matrix")
    matrix.add_argument("--execute", action="store_true")
    matrix.add_argument("--max-runs", type=int)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        import shutil

        from .models import ModelConfig, build_model
        from .runtime import environment_info, select_device

        result = environment_info(select_device())
        result["free_gib"] = round(shutil.disk_usage(".").free / 2**30, 2)
        result["parameters"] = {
            name: sum(p.numel() for p in build_model(ModelConfig(architecture=name)).parameters())
            for name in ("fixed", "global", "local", "cascade", "unet", "bilateral")
        }
    elif args.command == "acquire":
        from .acquisition import acquire_fivek

        result = acquire_fivek(
            args.output, args.full, args.train_count, args.val_count, args.test_count, args.workers
        )
    elif args.command == "verify-data":
        from .data import validate_manifest

        result = validate_manifest(args.manifest)
    elif args.command == "benchmark-inference":
        from .benchmarking import benchmark_inference

        result = benchmark_inference(args.checkpoint, args.image, args.output, args.device, args.repeats)
    elif args.command == "train":
        from .training import train

        result = train(_overrides(_config(args.config), args), resume=args.resume, initialize=args.initialize)
    elif args.command == "benchmark":
        from .training import benchmark

        result = benchmark(
            _overrides(_config(args.config), args),
            steps=args.steps,
            output_dir=args.output or "runs/benchmark",
        )
    elif args.command == "evaluate":
        from torch.utils.data import DataLoader

        from .data import PairedDataset, validate_manifest
        from .evaluation import evaluate
        from .runtime import file_sha256, load_checkpoint, select_device

        device = select_device(args.device)
        manifest_info = validate_manifest(args.manifest)
        model, payload = load_checkpoint(args.checkpoint, device)
        if not payload.get("metadata", {}).get("trained"):
            raise ValueError("Cannot report trained-model evaluation from an untrained checkpoint")
        for index, name in manifest_info["styles"].items():
            if index >= len(model.config.styles) or model.config.styles[index] != name:
                raise ValueError(f"Evaluation style {index}={name!r} does not match the checkpoint.")
        dataset = PairedDataset(args.manifest, args.split, image_size=args.image_size, limit=args.limit)
        protocol = {
            "split": args.split,
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "image_size": args.image_size,
            "seed": args.seed,
            "batch_size": 1,
            "training_metadata": payload.get("metadata", {}),
        }
        result = evaluate(
            model,
            DataLoader(dataset, batch_size=1),
            device,
            args.output,
            checkpoint=args.checkpoint,
            protocol=protocol,
            save_images=args.save_images,
            include_controls=args.controls,
            seed=args.seed,
        )
    elif args.command == "grade":
        from .inference import Grader

        grader = Grader(args.checkpoint, args.device)
        analysis = grader.analyze(args.image, args.style)
        result = grader.export(
            analysis, args.output, args.grade_strength, args.local_strength, args.treatment
        )
        if args.recipe:
            grader.save_recipe(analysis, args.recipe, args.treatment)
    elif args.command == "app":
        from .app import launch

        launch(
            args.checkpoint,
            args.device,
            port=args.port,
            export_dir=args.export_dir,
            raw_checkpoint=args.raw_checkpoint,
        )
        return
    elif args.command == "experiments":
        from .experiments import run_matrix

        result = run_matrix(args.matrix, execute=args.execute, max_runs=args.max_runs)
    else:
        parser.error("Unknown command")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
