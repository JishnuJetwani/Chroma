"""Command-line entry points for training and evaluation."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="photograder")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    acquire = sub.add_parser("acquire")
    acquire.add_argument("--output", default="data/fivek_pilot")
    acquire.add_argument("--full", action="store_true")
    verify = sub.add_parser("verify-data")
    verify.add_argument("manifest")
    for name in ("train", "benchmark"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
        item.add_argument("--manifest")
        item.add_argument("--device")
        item.add_argument("--output")
        item.add_argument("--max-steps", type=int)
        item.add_argument("--epochs", type=int)
        if name == "train":
            group = item.add_mutually_exclusive_group()
            group.add_argument("--initialize")
            group.add_argument("--resume")
        else:
            item.add_argument("--steps", type=int, default=200)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate.add_argument("--image-size", type=int, default=512)
    matrix = sub.add_parser("experiments")
    matrix.add_argument("matrix")
    matrix.add_argument("--execute", action="store_true")
    app = sub.add_parser("app")
    app.add_argument("--checkpoint", required=True)
    app.add_argument("--device", default="auto")
    app.add_argument("--port", type=int, default=7860)
    grade = sub.add_parser("grade")
    grade.add_argument("image")
    grade.add_argument("--checkpoint", required=True)
    grade.add_argument("--output", required=True)
    grade.add_argument("--device", default="auto")
    grade.add_argument("--style")
    grade.add_argument("--grade-strength", type=float, default=1)
    grade.add_argument("--local-strength", type=float, default=1)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from .runtime import environment_info, select_device

        result = environment_info(select_device())
    elif args.command == "acquire":
        from .acquisition import acquire_fivek

        result = acquire_fivek(args.output, full=args.full)
    elif args.command == "verify-data":
        from .data import validate_manifest

        result = validate_manifest(args.manifest)
    elif args.command in ("train", "benchmark"):
        from .training import benchmark, train

        config = json.loads(Path(args.config).read_text())
        if args.manifest:
            config.setdefault("data", {})["manifest"] = args.manifest
        if args.output:
            config["output_dir"] = args.output
        for key in ("device", "max_steps", "epochs"):
            if getattr(args, key, None) is not None:
                config.setdefault("training", {})[key] = getattr(args, key)
        result = (
            train(config, initialize=args.initialize, resume=args.resume)
            if args.command == "train"
            else benchmark(config, steps=args.steps, output_dir=args.output or "runs/benchmark")
        )
    elif args.command == "evaluate":
        from torch.utils.data import DataLoader

        from .data import PairedDataset, validate_manifest
        from .evaluation import evaluate
        from .runtime import load_checkpoint, select_device

        device = select_device(args.device)
        validate_manifest(args.manifest)
        model, payload = load_checkpoint(args.checkpoint, device)
        if not payload.get("metadata", {}).get("trained"):
            raise ValueError("An evaluated checkpoint must be trained")
        result = evaluate(
            model,
            DataLoader(PairedDataset(args.manifest, args.split, image_size=args.image_size), batch_size=1),
            device,
            args.output,
            checkpoint=args.checkpoint,
        )
    elif args.command == "experiments":
        from .experiments import run_matrix

        result = run_matrix(args.matrix, execute=args.execute)
    elif args.command == "app":
        from .app import launch

        launch(args.checkpoint, args.device, port=args.port)
        return
    elif args.command == "grade":
        from .inference import Grader

        grader = Grader(args.checkpoint, args.device)
        analysis = grader.analyze(args.image, args.style)
        result = grader.export(analysis, args.output, args.grade_strength, args.local_strength)
    print(json.dumps(result, indent=2, default=str))
