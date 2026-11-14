"""Action value container."""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Action:
    """A predicted action tensor and its interpretation.

    Args:
        values: Action tensor. A single-step action usually has shape ``(D,)``.
        representation: Action representation, such as ``"continuous"``.
        horizon: Number of consecutive time steps represented by ``values``.
    """

    values: torch.Tensor
    representation: str = "continuous"
    horizon: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if not self.representation.strip():
            raise ValueError("representation must not be empty")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
