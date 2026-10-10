"""Portable model output: color tables and image-aligned adjustment maps."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from torch import Tensor


@dataclass
class EditRecipe:
    luts: Tensor | None = None  # B x 3 x D x D x D; axes are R,G,B
    exposure: Tensor | None = None  # B x 1 x Hmap x Wmap, stops
    log_contrast: Tensor | None = None  # log2 contrast multiplier
    guide: Tensor | None = None  # unpadded sRGB luminance, same coordinates
    rgb_residual: Tensor | None = None
    bilateral_grid: Tensor | None = None  # B x 12 x D x Hgrid x Wgrid
    source_size: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stages: list[EditRecipe] | None = None

    def _map(self, fn) -> EditRecipe:
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Tensor):
                value = fn(value)
            elif item.name == "stages" and value is not None:
                value = [stage._map(fn) for stage in value]
            elif isinstance(value, dict):
                value = dict(value)
            values[item.name] = value
        return EditRecipe(**values)

    def to(self, *args, **kwargs) -> EditRecipe:
        return self._map(lambda tensor: tensor.to(*args, **kwargs))

    def detach(self) -> EditRecipe:
        return self._map(lambda tensor: tensor.detach())

    def state_dict(self) -> dict[str, Any]:
        """Tensor-only payload compatible with torch.load(weights_only=True)."""
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        if self.stages is not None:
            result["stages"] = [stage.state_dict() for stage in self.stages]
        return result

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> EditRecipe:
        state = dict(state)
        if state.get("stages") is not None:
            state["stages"] = [cls.from_state_dict(stage) for stage in state["stages"]]
        return cls(**state)
