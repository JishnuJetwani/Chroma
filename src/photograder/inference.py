"""Load once, analyze once, re-render and export without touching the source file."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps

from .recipes import EditRecipe
from .renderer import render
from .runtime import file_sha256, load_checkpoint, select_device, synchronize


def image_tensor(image: Image.Image, device="cpu") -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach()[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint(array * 255).astype(np.uint8))


@dataclass
class Analysis:
    source_path: str
    original: Image.Image
    preview: Image.Image
    recipe: EditRecipe
    style: str
    analysis_seconds: float


class Grader:
    def __init__(self, checkpoint: str | Path, device="auto"):
        self.device = select_device(device)
        self.model, self.checkpoint = load_checkpoint(checkpoint, self.device)
        if not self.checkpoint.get("metadata", {}).get("trained", False):
            raise ValueError(
                "This checkpoint has not been trained. Run a training experiment before grading."
            )
        self.checkpoint_path = str(Path(checkpoint).resolve())
        self.styles = list(self.model.config.styles)

    @torch.inference_mode()
    def analyze(self, path: str | Path, style=None) -> Analysis:
        from .data import load_srgb

        path = Path(path)
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise ValueError("Choose a JPEG or PNG photo.")
        with Image.open(path) as check:
            if getattr(check, "n_frames", 1) != 1:
                raise ValueError("Choose a still image; animated images are not supported.")
            if check.width * check.height > 60000000:
                raise ValueError("Please use a photo no larger than 60 megapixels.")
            if "A" in check.getbands() or "transparency" in check.info:
                alpha = ImageOps.exif_transpose(check).convert("RGBA").getchannel("A")
                if alpha.getextrema() != (255, 255):
                    raise ValueError(
                        "This photo has transparency. Flatten it to an opaque background before grading."
                    )
        original = load_srgb(path)
        if isinstance(original, tuple):
            original = original[0]
        style = style or self.styles[0]
        if style not in self.styles:
            raise ValueError(f"Unknown style: {style}")
        preview = original.copy()
        preview.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        source = image_tensor(preview, self.device)
        style_id = torch.tensor([self.styles.index(style)], device=self.device)
        synchronize(self.device)
        started = time.perf_counter()
        recipe = self.model.analyze(source, style_id).detach()
        synchronize(self.device)
        return Analysis(str(path.resolve()), original, preview, recipe, style, time.perf_counter() - started)

    @torch.inference_mode()
    def render_preview(self, analysis: Analysis, grade_strength=1.0, local_strength=1.0):
        source = image_tensor(analysis.preview, self.device)
        output = render(
            source,
            analysis.recipe,
            grade_strength=float(grade_strength),
            local_strength=float(local_strength),
            tile_size=512,
        )
        return tensor_image(output)

    @torch.inference_mode()
    def export(self, analysis: Analysis, destination: str | Path, grade_strength=1.0, local_strength=1.0):
        destination = Path(destination)
        if destination.resolve() == Path(analysis.source_path).resolve():
            raise ValueError("Export must use a new file; the original is preserved.")
        if destination.exists():
            raise FileExistsError(f"Export already exists: {destination}")
        if destination.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise ValueError("Export must end in .jpg or .png")
        source = image_tensor(analysis.original, self.device)
        started = time.perf_counter()
        result = render(
            source,
            analysis.recipe,
            grade_strength=float(grade_strength),
            local_strength=float(local_strength),
            tile_size=512,
        )
        output = tensor_image(result)
        destination.parent.mkdir(parents=True, exist_ok=True)
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        options = {"icc_profile": profile}
        if destination.suffix.lower() in (".jpg", ".jpeg"):
            options.update(quality=95, subsampling=0)
        output.save(destination, **options)
        return {
            "path": str(destination.resolve()),
            "width": output.width,
            "height": output.height,
            "seconds": time.perf_counter() - started,
            "style": analysis.style,
            "checkpoint": self.checkpoint_path,
        }

    def save_recipe(self, analysis: Analysis, destination):
        destination = Path(destination)
        if destination.exists() or destination.resolve() == Path(analysis.source_path).resolve():
            raise FileExistsError("Recipe export must use a new file; existing files are preserved.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "recipe": analysis.recipe.to("cpu").state_dict(),
                "style": analysis.style,
                "checkpoint": self.checkpoint_path,
                "checkpoint_sha256": file_sha256(self.checkpoint_path),
                "source_sha256": file_sha256(analysis.source_path),
                "source_dimensions": list(analysis.original.size),
            },
            destination,
        )


def adjustment_overlay(analysis: Analysis, name: str) -> Image.Image:
    image = np.asarray(analysis.preview).astype(np.float32) / 255
    field = getattr(analysis.recipe, name, None)
    if field is None and analysis.recipe.stages:
        field = next(
            (
                getattr(stage, name)
                for stage in analysis.recipe.stages
                if getattr(stage, name, None) is not None
            ),
            None,
        )
    if field is None:
        return analysis.preview.copy()
    data = field.detach().float()[0, 0].cpu().numpy()
    data = np.asarray(
        Image.fromarray(data, mode="F").resize(analysis.preview.size, Image.Resampling.BILINEAR)
    )
    magnitude = np.clip(abs(data), 0, 1)[..., None] * 0.65
    positive = np.array([1.0, 0.48, 0.12])
    negative = np.array([0.13, 0.55, 1.0])
    color = np.where((data > 0)[..., None], positive, negative)
    return Image.fromarray(np.rint((image * (1 - magnitude) + color * magnitude) * 255).astype(np.uint8))
