"""Portable model output: color tables and image-aligned adjustment maps."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from torch import Tensor


@dataclass
class EditRecipe:
    luts: Tensor | None = None
    exposure: Tensor | None = None
    log_contrast: Tensor | None = None
    guide: Tensor | None = None
    source_size: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def _map(self, fn) -> EditRecipe:
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Tensor):
                value = fn(value)
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
        return result

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> EditRecipe:
        state = dict(state)
        return cls(**state)
