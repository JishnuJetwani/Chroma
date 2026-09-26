"""Small style-conditioned CNNs that predict executable photo edit recipes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .recipes import EditRecipe
from .renderer import identity_lut, render


@dataclass
class ModelConfig:
    architecture: str = "global"
    styles: tuple[str, ...] = ("Expert C",)
    preview_size: int = 256
    lut_size: int = 33
    num_luts: int = 8

    def __post_init__(self):
        self.styles = tuple(self.styles)
        if self.architecture not in ("fixed", "global"):
            raise ValueError("Unknown architecture: " + self.architecture)
        if not self.styles or len(set(self.styles)) != len(self.styles):
            raise ValueError("styles must be a nonempty sequence of unique names")
        if self.preview_size < 32 or self.lut_size < 2 or self.num_luts < 1:
            raise ValueError("Invalid preview or LUT size")


def _preview(image: Tensor, size: int):
    height, width = image.shape[-2:]
    scale = size / max(height, width)
    new_h, new_w = (max(1, round(height * scale)), max(1, round(width * scale)))
    resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top, left = ((size - new_h) // 2, (size - new_w) // 2)
    return F.pad(resized, (left, size - new_w - left, top, size - new_h - top), mode="replicate")


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvBlock(3, 32, 2), ConvBlock(32, 64, 2), ConvBlock(64, 128, 2), ConvBlock(128, 128, 2)]
        )

    def forward(self, image):
        for block in self.blocks:
            image = block(image)
        return image


class LUTBank(nn.Module):
    def __init__(self, count, size):
        super().__init__()
        identity = identity_lut(size)
        values = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3, 1, 1, 1)
        coefficients = torch.sin(values * 2.39996 + 0.3)
        coefficients -= coefficients.mean(dim=0, keepdim=True)
        variation = 0.02 * coefficients * identity[None] * (1 - identity[None])
        self.tables = nn.Parameter(identity[None].repeat(count, 1, 1, 1, 1) + variation)

    def forward(self, weights):
        return torch.einsum("bn,ncijk->bcijk", weights, self.tables)

    def regularization(self):
        smooth, monotonic = (self.tables.new_zeros(()), self.tables.new_zeros(()))
        for axis in (2, 3, 4):
            diff = self.tables.diff(dim=axis)
            monotonic = monotonic + F.relu(-diff).square().mean()
            if diff.shape[axis] > 1:
                smooth = smooth + diff.diff(dim=axis).square().mean()
        return {"lut_smoothness": smooth, "lut_monotonicity": monotonic}


class RecipeModel(nn.Module):
    def forward(self, image: Tensor, style: Tensor | None = None) -> Tensor:
        return render(image, self.analyze(image, style))

    def _style(self, image, style):
        if style is None:
            style = torch.zeros(image.shape[0], dtype=torch.long, device=image.device)
        elif isinstance(style, int):
            style = torch.full((image.shape[0],), style, dtype=torch.long, device=image.device)
        else:
            style = style.to(device=image.device, dtype=torch.long)
        if style.shape != (image.shape[0],):
            raise ValueError("style must contain one index per image")
        valid = ((style >= 0) & (style < len(self.config.styles))).all()
        if image.device.type == "cuda":
            torch._assert_async(valid, "style index outside configured styles")
        elif not bool(valid):
            raise ValueError("style index outside configured styles")
        return style

    def _metadata(self):
        return {"version": 1, "architecture": self.config.architecture}


class FixedLUTModel(RecipeModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bank = LUTBank(len(config.styles), config.lut_size)
        with torch.no_grad():
            self.bank.tables.copy_(identity_lut(config.lut_size)[None])

    def analyze(self, image, style=None):
        style = self._style(image, style)
        return EditRecipe(
            luts=self.bank.tables[style], source_size=tuple(image.shape[-2:]), metadata=self._metadata()
        )

    def regularization(self):
        return self.bank.regularization()


class GradingModel(RecipeModel):
    """Predict image-dependent mixing weights over learned global color tables."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder()
        self.style_embedding = nn.Embedding(len(config.styles), 16)
        nn.init.normal_(self.style_embedding.weight, std=0.02)
        self.bank = LUTBank(config.num_luts, config.lut_size)
        self.mixer = nn.Linear(128 + 16, config.num_luts)
        nn.init.zeros_(self.mixer.weight)
        nn.init.zeros_(self.mixer.bias)

    def analyze(self, image, style=None):
        style = self._style(image, style)
        preview = _preview(image, self.config.preview_size)
        features = self.encoder(preview)
        pooled = features.float().mean((-2, -1))
        weights = self.mixer(torch.cat((pooled, self.style_embedding(style)), 1)).softmax(-1)
        return EditRecipe(
            luts=self.bank(weights), source_size=tuple(image.shape[-2:]), metadata=self._metadata()
        )

    def regularization(self):
        return self.bank.regularization()


def build_model(config=None):
    config = config or ModelConfig()
    if isinstance(config, dict):
        config = ModelConfig(**config)
    return FixedLUTModel(config) if config.architecture == "fixed" else GradingModel(config)


def initialize_from_model(target: RecipeModel, source: RecipeModel) -> dict:
    """Transfer compatible weights, matching learned style rows by name.

    Global LUT bank indices denote shared basis functions, not editor IDs.
    Only fixed LUTs and explicit style embeddings have a style-indexed axis.
    Newly introduced styles retain their fresh initialization. This helper is
    for initialization of a new experiment, never optimizer-state resumption.
    """

    def style_parameters(model):
        result = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and name.split(".")[-1] == "style_embedding":
                result.add(name + ".weight")
            if isinstance(module, FixedLUTModel):
                result.add((name + "." if name else "") + "bank.tables")
        return result

    target_state, source_state = (target.state_dict(), source.state_dict())
    target_style_keys, source_style_keys = (style_parameters(target), style_parameters(source))
    source_indices = {name: index for index, name in enumerate(source.config.styles)}
    matching = [
        (index, source_indices[name], name)
        for index, name in enumerate(target.config.styles)
        if name in source_indices
    ]
    compatible, copied_rows, skipped = ({}, {}, [])
    for key, value in source_state.items():
        if key not in target_state:
            skipped.append(key)
            continue
        destination = target_state[key]
        is_style = key in source_style_keys or key in target_style_keys
        if is_style:
            if (
                key not in source_style_keys
                or key not in target_style_keys
                or destination.shape[1:] != value.shape[1:]
                or (not matching)
            ):
                skipped.append(key)
                continue
            remapped = destination.detach().clone()
            for new_index, old_index, _ in matching:
                remapped[new_index].copy_(
                    value[old_index].to(device=destination.device, dtype=destination.dtype)
                )
            compatible[key] = remapped
            copied_rows[key] = [
                {"style": name, "source_index": old, "target_index": new} for new, old, name in matching
            ]
        elif value.shape == destination.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    if not compatible:
        raise ValueError("No compatible weights in initialization checkpoint.")
    target.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": len(compatible),
        "style_rows_copied": copied_rows,
        "new_styles": [name for name in target.config.styles if name not in source_indices],
        "source_styles": list(source.config.styles),
        "target_styles": list(target.config.styles),
        "skipped_tensors": skipped,
        "fresh_target_tensors": [key for key in target_state if key not in compatible],
    }
