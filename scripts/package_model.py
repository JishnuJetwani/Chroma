"""Package the recorded JPEG model without including datasets or private run metadata."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from photograder.runtime import file_sha256, load_checkpoint

JPEG_SHA256 = "1e60aaab4c95a9c3755a6d6f314c4c531b355d984dc8edab5c8f776c5e8d657a"


def package_model(checkpoint: Path, destination: Path) -> Path:
    """Validate the documented weights and write a standalone, explicitly local bundle."""
    checkpoint, destination = Path(checkpoint), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if destination.suffix.lower() != ".zip":
        raise ValueError("Model bundle must use the .zip extension")
    digest = file_sha256(checkpoint)
    if digest != JPEG_SHA256:
        raise ValueError("Checkpoint does not match the published model card checksum")
    model, payload = load_checkpoint(checkpoint, "cpu")
    if not payload.get("metadata", {}).get("trained"):
        raise ValueError("Cannot package an untrained model")
    metadata = {
        "format_version": 1,
        "checkpoint": "checkpoints/jpeg.pt",
        "sha256": digest,
        "model_config": payload["model_config"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "run": "full-7030-20261012",
        "terms": "No commercial-use permission granted. See the model card and FiveK dataset terms.",
    }
    model_card = Path(__file__).resolve().parents[1] / "docs/model-card.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-bundle-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "bundle.zip"
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(checkpoint, "checkpoints/jpeg.pt")
            archive.write(model_card, "docs/model-card.md")
            archive.writestr("checkpoints/jpeg.json", json.dumps(metadata, indent=2) + "\n")
            archive.writestr("checkpoints/SHA256SUMS", f"{digest}  jpeg.pt\n")
        # Exclusive creation also protects against a destination appearing mid-build.
        with destination.open("xb") as output, staged.open("rb") as source:
            import shutil

            shutil.copyfileobj(source, output)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(package_model(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
