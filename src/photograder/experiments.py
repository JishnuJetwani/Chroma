"""Explicit experiment dispatch and blinded, scene-aware human evaluation."""

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .metrics import scene_bootstrap


def run_matrix(
    matrix_path, execute: bool = False, max_runs: int | None = None, python_executable: str | None = None
) -> list:
    """Dry-run by default. Execution is sequential on the selected local device."""
    path = Path(matrix_path).resolve()
    matrix = json.loads(path.read_text())
    entries = matrix.get("experiments", [])
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive")
    selected = [entry for entry in entries if entry.get("enabled", True)]
    if max_runs is not None:
        selected = selected[:max_runs]
    plan = []
    working_directory = (
        (path.parent / matrix["working_directory"]).resolve()
        if matrix.get("working_directory")
        else Path.cwd()
    )
    if not working_directory.is_dir():
        raise FileNotFoundError(working_directory)
    for entry in selected:
        config = (path.parent / entry["config"]).resolve()
        if not config.is_file():
            raise FileNotFoundError(config)
        command = [python_executable or sys.executable, "-m", "photograder", "train", "--config", str(config)]
        if entry.get("initialize"):
            command.extend(["--initialize", str((path.parent / entry["initialize"]).resolve())])
        plan.append(
            {
                "id": entry["id"],
                "command": command,
                "working_directory": str(working_directory),
                "executed": False,
            }
        )
    # Validate the complete selected matrix before starting its first mutation.
    if execute:
        for item in plan:
            completed = subprocess.run(item["command"], cwd=item["working_directory"], check=True)
            item["executed"] = True
            item["returncode"] = completed.returncode
    return plan


def compare_evaluations(
    report_paths: Sequence[str], output_dir, seed: int = 42, bootstrap_samples: int = 2000
) -> dict:
    """Compare matched reports, bootstrapping paired candidate-minus-baseline deltas.

    The first report is the reference. This validates evaluation comparability,
    not equal training budgets or external checkpoint provenance.
    """
    if len(report_paths) < 2:
        raise ValueError("Supply a reference report followed by at least one candidate")
    loaded = []
    for supplied in report_paths:
        path = Path(supplied)
        if path.is_dir():
            path = path / "metrics.json"
        report = json.loads(path.read_text())
        with path.with_name("metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != report["images"]:
            raise ValueError("Report/image CSV count mismatch: " + str(path))
        keys = [(row["scene_id"], row["style_id"], row["image_index"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate evaluation record IDs")
        loaded.append((path, report, dict(zip(keys, rows, strict=False))))
    reference_path, reference, reference_rows = loaded[0]
    required_protocol = ("manifest_sha256", "split", "image_size")
    for name in required_protocol:
        if reference.get("protocol", {}).get(name) is None:
            raise ValueError("Reference lacks required evaluation protocol field: " + name)
    comparisons = []
    for path, report, rows in loaded[1:]:
        for name in required_protocol + ("seed",):
            if report.get("protocol", {}).get(name) != reference["protocol"].get(name):
                raise ValueError(f"Evaluation protocol differs for {name}: {path}")
        for name in ("metric_note", "metric_version", "torch_version", "controls", "device"):
            if report.get(name) != reference.get(name):
                raise ValueError(f"Evaluation {name} differs: {path}")
        ref_styles = (reference.get("model_config") or {}).get("styles")
        styles = (report.get("model_config") or {}).get("styles")
        if ref_styles != styles:
            raise ValueError("Style registries differ")
        if set(rows) != set(reference_rows):
            raise ValueError("Evaluation image/scene/style records differ")
        metric_names = sorted(set(reference["metrics"]) & set(report["metrics"]))
        deltas = {}
        for name in metric_names:
            if name.startswith("input_") or name == "inference_seconds_per_image":
                continue
            differences, scenes = [], []
            for identity in sorted(reference_rows):
                left, right = reference_rows[identity].get(name, ""), rows[identity].get(name, "")
                if left == "" or right == "":
                    continue
                difference = float(right) - float(left)
                if np.isfinite(difference):
                    differences.append(difference)
                    scenes.append(identity[0])
            deltas[name] = scene_bootstrap(differences, scenes, seed, bootstrap_samples)
        comparisons.append(
            {
                "candidate_report": str(path.resolve()),
                "checkpoint": report.get("checkpoint"),
                "checkpoint_sha256": report.get("checkpoint_sha256"),
                "paired_deltas": deltas,
            }
        )
    result = {
        "reference_report": str(reference_path.resolve()),
        "reference_checkpoint": reference.get("checkpoint"),
        "protocol": {key: reference["protocol"][key] for key in required_protocol},
        "images": reference["images"],
        "comparisons": comparisons,
        "interpretation": "All deltas are candidate minus reference. Positive PSNR/SSIM is better; negative MAE/MSE/CIE76 and outside drift is better. Clipping fractions are diagnostic. Paired scene-bootstrap intervals reflect sampled scenes, not training-seed uncertainty. This does not establish equal training budgets or novelty.",
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Matched evaluation comparison",
        "",
        f"Reference: `{reference_path}`",
        "",
        result["interpretation"],
        "",
        "| Candidate report | Metric | Paired delta | 95% scene bootstrap CI |",
        "|---|---|---:|---:|",
    ]
    for comparison in comparisons:
        for name, values in comparison["paired_deltas"].items():
            if values["mean"] is not None:
                lines.append(
                    "| {} | {} | {:.6g} | [{:.6g}, {:.6g}] |".format(
                        comparison["candidate_report"],
                        name,
                        values["mean"],
                        values["ci_low"],
                        values["ci_high"],
                    )
                )
    (destination / "comparison.md").write_text("\n".join(lines) + "\n")
    return result


def _panel(path, max_edge):
    with Image.open(path) as source:
        panel = source.convert("RGB")
        panel.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return panel.copy()


def create_blind_comparisons(
    records: Sequence[dict],
    output_dir,
    model_a: str = "baseline",
    model_b: str = "candidate",
    seed: int = 42,
    max_edge: int = 900,
) -> dict:
    """Records need scene_id, image_id, input_path, a_path, b_path.

    Share sheets/ and votes.csv, NEVER the private key/model metadata with raters.
    Each rating row additionally records rater_id; duplicate ratings are rejected.
    """
    destination = Path(output_dir)
    sheets = destination / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    if model_a == model_b:
        raise ValueError("Models must have distinct names")
    if len({str(record["image_id"]) for record in records}) != len(records):
        raise ValueError("image_id values must be unique")
    rng = random.Random(seed)
    ordered = list(records)
    rng.shuffle(ordered)
    key_rows, vote_rows = [], []
    for index, record in enumerate(ordered):
        flip = rng.random() < 0.5
        left_name, right_name = (model_b, model_a) if flip else (model_a, model_b)
        left_path, right_path = (
            (record["b_path"], record["a_path"]) if flip else (record["a_path"], record["b_path"])
        )
        panels = [_panel(p, max_edge) for p in (record["input_path"], left_path, right_path)]
        if panels[1].size != panels[2].size:
            raise ValueError("Candidate outputs must have identical display dimensions")
        width = max(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (width * 3, height + 32), (32, 32, 32))
        draw = ImageDraw.Draw(canvas)
        comparison_id = f"comparison-{index:04d}"
        for number, (panel, label) in enumerate(zip(panels, ("Original", "A", "B"), strict=False)):
            canvas.paste(panel, (number * width + (width - panel.width) // 2, 32))
            draw.text((number * width + 8, 9), label, fill="white")
        canvas.save(sheets / (comparison_id + ".png"))
        key_rows.append(
            {
                "comparison_id": comparison_id,
                "scene_id": str(record["scene_id"]),
                "image_id": str(record["image_id"]),
                "left_model": left_name,
                "right_model": right_name,
            }
        )
        vote_rows.append(
            {
                "comparison_id": comparison_id,
                "rater_id": "",
                "overall": "",
                "local_correction": "",
                "fewer_artifacts": "",
            }
        )
    for name, values in (("private_key.csv", key_rows), ("votes.csv", vote_rows)):
        fields = (
            list(values[0])
            if values
            else (
                ["comparison_id", "scene_id", "image_id", "left_model", "right_model"]
                if name == "private_key.csv"
                else ["comparison_id", "rater_id", "overall", "local_correction", "fewer_artifacts"]
            )
        )
        with (destination / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    metadata = {
        "seed": seed,
        "comparisons": len(records),
        "model_a": model_a,
        "model_b": model_b,
        "instructions": "Keep private_key.csv private. Raters choose A, B or tie independently for overall preference, correction of the faulty area, and fewer visible artifacts. Blank means unanswered. Duplicate the template for multiple raters, set stable anonymous rater_id, combine rows for analysis.",
    }
    (destination / "private_protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def analyze_votes(
    votes_path,
    key_path,
    candidate: str = "candidate",
    seed: int = 42,
    bootstrap_samples: int = 2000,
    output_path=None,
) -> dict:
    with Path(key_path).open(newline="") as handle:
        key_rows = list(csv.DictReader(handle))
    key = {row["comparison_id"]: row for row in key_rows}
    if len(key) != len(key_rows):
        raise ValueError("Duplicate comparison IDs in private key")
    with Path(votes_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = {}
    for outcome in ("overall", "local_correction", "fewer_artifacts"):
        score_by_image, scene_by_image = {}, {}
        counts = {"wins": 0, "losses": 0, "ties": 0, "unanswered": 0}
        seen = set()
        for row in rows:
            choice = row.get(outcome, "").strip().lower()
            if not choice:
                counts["unanswered"] += 1
                continue
            identifier = row["comparison_id"]
            if identifier not in key:
                raise ValueError("Unknown comparison ID: " + identifier)
            rater = row.get("rater_id", "").strip()
            if not rater:
                raise ValueError("Every answered row must have an anonymous rater_id")
            identity = (identifier, rater)
            if identity in seen:
                raise ValueError("Duplicate rater/comparison vote: " + str(identity))
            seen.add(identity)
            if choice not in ("a", "b", "tie"):
                raise ValueError("Votes must be A, B, tie, or blank")
            info = key[identifier]
            if candidate not in (info["left_model"], info["right_model"]):
                raise ValueError("Candidate does not participate in comparison " + identifier)
            if choice == "tie":
                counts["ties"] += 1
                continue
            picked = info["left_model"] if choice == "a" else info["right_model"]
            score = float(picked == candidate)
            counts["wins" if score else "losses"] += 1
            image_id = info["image_id"]
            score_by_image.setdefault(image_id, []).append(score)
            scene_by_image[image_id] = info["scene_id"]
        image_ids = list(score_by_image)
        statistics = scene_bootstrap(
            [float(np.mean(score_by_image[i])) for i in image_ids],
            [scene_by_image[i] for i in image_ids],
            seed,
            bootstrap_samples,
        )
        statistics.update(counts)
        statistics["decisive_vote_win_rate"] = (
            counts["wins"] / (counts["wins"] + counts["losses"])
            if counts["wins"] + counts["losses"]
            else None
        )
        results[outcome] = statistics
    result = {
        "candidate": candidate,
        "outcomes": results,
        "protocol": "Ties reported separately, excluded from decisive preference. Ratings averaged by image, then by scene; scenes resampled. These intervals condition on the recruited raters and do not quantify the full population of raters. Do not treat repeated ratings as independent photos.",
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def run_external_deeplpf(
    repo_path, expected_commit: str, command: Sequence[str], output_dir, protocol: dict
) -> dict:
    """Run an explicit external-environment command only at an audited commit.

    This adapter does not install dependencies, download weights or certify the
    external model's training split. The caller supplies its documented CLI.
    """
    if len(expected_commit) != 40 or any(c not in "0123456789abcdef" for c in expected_commit.lower()):
        raise ValueError("Pin the full 40-character audited DeepLPF git commit")
    repository = Path(repo_path).resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if actual.lower() != expected_commit.lower():
        raise ValueError("External repository differs from the audited commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip():
        raise ValueError("External baseline checkout must be clean for provenance")
    if not command:
        raise ValueError("Supply the explicit external Python/CLI command")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(list(command), cwd=repository, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    (destination / "external_stdout.log").write_text(completed.stdout)
    (destination / "external_stderr.log").write_text(completed.stderr)
    report = {
        "repository": str(repository),
        "commit": actual,
        "command": list(command),
        "protocol": protocol,
        "process_wall_seconds": elapsed,
        "timing_note": "Entire external process including startup, checkpoint loading, input/output; not model-only latency.",
        "returncode": completed.returncode,
        "status": "external execution completed; reproduction and overlap audit remain separate claims"
        if completed.returncode == 0
        else "external command failed; inspect logs",
    }
    (destination / "external_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    completed.check_returncode()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    matrix = commands.add_parser("matrix", help="Print explicit training commands; --execute opts in")
    matrix.add_argument("path")
    matrix.add_argument("--execute", action="store_true")
    matrix.add_argument("--max-runs", type=int)
    blind = commands.add_parser("blind", help="Create randomized comparison PNGs and vote template")
    blind.add_argument("records", help="JSON array with scene_id/image_id/input_path/a_path/b_path")
    blind.add_argument("output_dir")
    blind.add_argument("--seed", type=int, default=42)
    votes = commands.add_parser("votes", help="Analyze anonymous CSV votes using private key")
    votes.add_argument("votes_path")
    votes.add_argument("key_path")
    votes.add_argument("--candidate", default="candidate")
    votes.add_argument("--output")
    compare = commands.add_parser(
        "compare", help="Compare matched evaluation reports with paired scene intervals"
    )
    compare.add_argument("reports", nargs="+")
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.action == "matrix":
        result = run_matrix(args.path, args.execute, args.max_runs)
    elif args.action == "blind":
        result = create_blind_comparisons(
            json.loads(Path(args.records).read_text()), args.output_dir, seed=args.seed
        )
    elif args.action == "votes":
        result = analyze_votes(args.votes_path, args.key_path, args.candidate, output_path=args.output)
    else:
        result = compare_evaluations(args.reports, args.output_dir, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
