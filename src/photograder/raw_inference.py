"""Private RAW analysis and cached, memory-bounded native-resolution export."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageCms
from torch.nn import functional as F

from .inference import tensor_image
from .raw_io import RAW_PIPELINE_VERSION, decode_raw
from .raw_model import (
    RawEditRecipe,
    RawGradingModel,
    RawModelConfig,
    baseline_raw,
    render_raw,
    render_raw_tile,
)
from .runtime import file_sha256, select_device, synchronize

RAW_EXTENSIONS = (
    ".dng",
    ".nef",
    ".nrw",
    ".cr2",
    ".cr3",
    ".arw",
    ".srf",
    ".sr2",
    ".raf",
    ".rw2",
    ".orf",
    ".pef",
    ".rwl",
    ".3fr",
    ".fff",
    ".iiq",
    ".kdc",
    ".dcr",
    ".mos",
    ".mrw",
    ".raw",
)
DEFAULT_RAW_CHECKPOINT = Path("runs/raw-pilot-20261030/shared/model.pt")
JPEG_CHECKPOINT_CANDIDATES = (
    Path("checkpoints/jpeg.pt"),
    Path("runs/full-7030-20261012/joint/model.pt"),
    Path("runs/multistyle_pilot/best.pt"),
    Path("runs/local_pilot/best.pt"),
)


def is_raw_checkpoint(path: str | Path) -> bool:
    """Inspect local tensor-only checkpoint metadata without constructing a model."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return isinstance(payload, dict) and "raw_pipeline_version" in payload


def load_raw_checkpoint(path: str | Path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload or "model_config" not in payload:
        raise ValueError("Expected a RAW grading checkpoint containing model and model_config.")
    if payload.get("raw_pipeline_version") != RAW_PIPELINE_VERSION:
        raise ValueError(
            "This RAW checkpoint uses a different decoding pipeline. Use its matching decoder/checkpoint."
        )
    if not payload.get("metadata", {}).get("trained", False):
        raise ValueError("This RAW checkpoint has not been trained. Complete training before grading.")
    model = RawGradingModel(RawModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


@dataclass
class RawAnalysis:
    source_path: str
    linear_rgb: torch.Tensor  # Native B3HW view of CPU float32 HWC storage
    preview_linear: torch.Tensor  # CPU B3HW; never encoded or quantized
    preview: Image.Image  # Neutral display development for comparison
    recipe: RawEditRecipe  # CPU cache, transferable per render
    style: str
    analysis_seconds: float
    raw_metadata: dict

    @property
    def source_size(self):
        return tuple(self.linear_rgb.shape[-2:])


class RawGrader:
    def __init__(self, checkpoint: str | Path, device="auto", *, preview_size=1024, tile_size=512):
        if not isinstance(preview_size, int) or preview_size < 32:
            raise ValueError("preview_size must be an integer of at least 32")
        if not isinstance(tile_size, int) or tile_size < 1:
            raise ValueError("tile_size must be a positive integer")
        self.device = select_device(device)
        self.model, self.checkpoint = load_raw_checkpoint(checkpoint, self.device)
        self.checkpoint_path = str(Path(checkpoint).resolve())
        self.styles = list(self.model.config.styles)
        self.preview_size, self.tile_size = preview_size, tile_size

    @torch.inference_mode()
    def analyze(self, path: str | Path, style=None) -> RawAnalysis:
        path = Path(path)
        if path.suffix.lower() not in RAW_EXTENSIONS:
            raise ValueError("Choose a camera RAW file, such as DNG, RAF, NEF, ARW, or CR2.")
        style = style or self.styles[0]
        if style not in self.styles:
            raise ValueError(f"Unknown style: {style}")
        started = time.perf_counter()
        if path.suffix.lower() == ".dng":
            from .raw_acquisition import dng_crop_metadata
            from .raw_data import crop_dng_raw

            geometry = dng_crop_metadata(path)
            decoded = decode_raw(path, max_pixels=60_000_000, apply_orientation=False)
            cropped, crop_audit = crop_dng_raw(
                decoded.linear_rgb, dng_geometry=geometry, decoder_metadata=decoded.metadata
            )
            array = cropped
            raw_metadata = {**decoded.metadata, "development_geometry": crop_audit}
        else:
            decoded = decode_raw(path, max_pixels=60_000_000)
            array, raw_metadata = decoded.linear_rgb, dict(decoded.metadata)
        if array.dtype != np.float32 or array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("RAW decoder must return float32 HWC RGB.")
        if not np.isfinite(array).all() or min(array.shape) < 1:
            raise ValueError("RAW decoder returned invalid pixels.")
        if array.shape[0] * array.shape[1] > 60_000_000:
            raise ValueError("Please use a RAW photo no larger than 60 megapixels.")
        if decoded.metadata.get("pipeline_version") != RAW_PIPELINE_VERSION:
            raise ValueError("RAW decoder metadata does not match the trained pipeline.")
        # The native image never moves to the accelerator. This permutation is
        # a view, preserving the unclipped decoder samples without a second copy.
        source = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        scale = min(1.0, self.preview_size / max(array.shape[:2]))
        size = tuple(max(1, round(value * scale)) for value in array.shape[:2])
        preview = F.interpolate(source, size=size, mode="bilinear", align_corners=False, antialias=True)
        analysis_scale = self.model.config.preview_size / max(array.shape[:2])
        analysis_size = tuple(max(1, round(value * analysis_scale)) for value in array.shape[:2])
        # Match dataset preparation directly from native scene-linear samples,
        # avoiding a second resampling through the separate display preview.
        analysis_source = F.interpolate(
            source, size=analysis_size, mode="bilinear", align_corners=False, antialias=True
        ).to(self.device)
        recipe = self.model.analyze(analysis_source, self.styles.index(style)).detach().to("cpu")
        recipe.source_size = tuple(array.shape[:2])
        baseline = tensor_image(baseline_raw(preview.to(self.device)))
        synchronize(self.device)
        return RawAnalysis(
            str(path.resolve()),
            source,
            preview,
            baseline,
            recipe,
            style,
            time.perf_counter() - started,
            raw_metadata,
        )

    @torch.inference_mode()
    def render_preview(self, analysis: RawAnalysis, grade_strength=1.0, exposure_bias_ev=0.0):
        output = render_raw(
            analysis.preview_linear.to(self.device),
            analysis.recipe,
            strength=float(grade_strength),
            exposure_bias_ev=float(exposure_bias_ev),
            tile_size=self.tile_size,
        )
        return tensor_image(output)

    @torch.inference_mode()
    def export(
        self, analysis: RawAnalysis, destination: str | Path, grade_strength=1.0, exposure_bias_ev=0.0
    ):
        destination = Path(destination)
        if destination.resolve() == Path(analysis.source_path).resolve():
            raise ValueError("Export must use a new file; the original is preserved.")
        if destination.exists():
            raise FileExistsError(f"Export already exists: {destination}")
        extension = destination.suffix.lower()
        if extension not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            raise ValueError("Export must end in .png, .jpg, or .tif")
        is_tiff = extension in (".tif", ".tiff")
        if is_tiff:
            try:
                import tifffile
            except ImportError as error:
                raise ValueError("16-bit TIFF export requires tifffile from the RAW dependencies.") from error
        started = time.perf_counter()
        height, width = analysis.source_size
        quantization = 65535 if is_tiff else 255
        pixels = np.empty((height, width, 3), dtype=np.uint16 if is_tiff else np.uint8)
        recipe = analysis.recipe.to(self.device)
        for top in range(0, height, self.tile_size):
            for left in range(0, width, self.tile_size):
                tile = analysis.linear_rgb[..., top : top + self.tile_size, left : left + self.tile_size].to(
                    self.device
                )
                rendered = render_raw_tile(
                    tile,
                    recipe,
                    full_size=(height, width),
                    origin=(top, left),
                    strength=float(grade_strength),
                    exposure_bias_ev=float(exposure_bias_ev),
                )
                # A whole floating-point output is never allocated on either
                # accelerator or CPU; only this tile and the final integer image.
                result = rendered[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                pixels[top : top + tile.shape[-2], left : left + tile.shape[-1]] = np.rint(
                    result * quantization
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        if is_tiff:
            tifffile.imwrite(
                destination,
                pixels,
                photometric="rgb",
                metadata=None,
                extratags=[(34675, "B", len(profile), profile, False)],
            )
        else:
            options = {"icc_profile": profile}
            if extension in (".jpg", ".jpeg"):
                options.update(quality=95, subsampling=0)
            Image.fromarray(pixels).save(destination, **options)
        return {
            "path": str(destination.resolve()),
            "width": width,
            "height": height,
            "seconds": time.perf_counter() - started,
            "style": analysis.style,
            "checkpoint": self.checkpoint_path,
            "pipeline_version": RAW_PIPELINE_VERSION,
            "bit_depth": 16 if is_tiff else 8,
            "color_space": "display-sRGB",
            "grade_strength": float(grade_strength),
            "exposure_bias_ev": float(exposure_bias_ev),
        }

    def save_recipe(self, analysis: RawAnalysis, destination):
        destination = Path(destination)
        if destination.exists() or destination.resolve() == Path(analysis.source_path).resolve():
            raise FileExistsError("Recipe export must use a new file; existing files are preserved.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "raw_pipeline_version": RAW_PIPELINE_VERSION,
                "recipe": analysis.recipe.to("cpu").state_dict(),
                "style": analysis.style,
                "checkpoint": self.checkpoint_path,
                "checkpoint_sha256": file_sha256(self.checkpoint_path),
                "source_sha256": file_sha256(analysis.source_path),
                "source_dimensions": list(analysis.source_size),
                "raw_metadata": analysis.raw_metadata,
            },
            destination,
        )
