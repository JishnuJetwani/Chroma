"""Portable global color-table predictions."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class EditRecipe:
    luts: Tensor | None = None
    source_size: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, *args, **kwargs):
        return EditRecipe(
            None if self.luts is None else self.luts.to(*args, **kwargs),
            self.source_size,
            dict(self.metadata),
        )

    def detach(self):
        return EditRecipe(
            None if self.luts is None else self.luts.detach(), self.source_size, dict(self.metadata)
        )

    def state_dict(self):
        return dict(luts=self.luts, source_size=self.source_size, metadata=self.metadata)

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)
