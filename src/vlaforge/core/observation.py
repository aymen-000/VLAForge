"""Input observation structure."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class Observation:
    """Robot observation containing one or more camera images and an instruction."""

    images: dict[str, Any]
    instruction: str
    proprioception: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.images:
            raise ValueError("images must contain at least one camera observation")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
