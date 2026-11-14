"""Batched model inputs and target actions."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class VLABatch:
    """A collated batch consumed by a VLA model.

    Images are keyed by camera name and typically have shape ``(B, C, H, W)``.
    Token tensors have shape ``(B, L)`` and target actions ``(B, D)`` for the
    single-step V1 task.
    """

    images: dict[str, torch.Tensor]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    actions: torch.Tensor | None = None
    proprioception: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.images:
            raise ValueError("images must contain at least one camera batch")
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch_size, sequence_length)")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        if self.actions is not None and self.actions.ndim < 2:
            raise ValueError("actions must include a batch dimension")

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "VLABatch":
        """Return a copy with all model tensors moved to ``device``."""
        return VLABatch(
            images={
                name: tensor.to(device=device, non_blocking=non_blocking)
                for name, tensor in self.images.items()
            },
            input_ids=self.input_ids.to(device=device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device=device, non_blocking=non_blocking),
            actions=(
                self.actions.to(device=device, non_blocking=non_blocking)
                if self.actions is not None
                else None
            ),
            proprioception=(
                self.proprioception.to(device=device, non_blocking=non_blocking)
                if self.proprioception is not None
                else None
            ),
            metadata=self.metadata,
        )
