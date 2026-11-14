"""Offline dataset evaluation for any ``BaseVLA`` model."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from vlaforge.core import BaseVLA, VLABatch
from vlaforge.data import ActionNormalizer, VLACollator
from vlaforge.inference.pipeline import resolve_device


class Evaluator:
    """Evaluate action predictions on a held-out dataset.

    Metrics are computed on normalized values by default. Set
    ``metric_space="physical"`` and pass the training-set ``ActionNormalizer``
    to compare predictions and targets in their original units.
    """

    def __init__(
        self,
        model: BaseVLA,
        dataset: Dataset[Any],
        collator: Callable[[list[Any]], VLABatch] | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        device: str | torch.device = "auto",
        metric_space: str = "normalized",
        action_normalizer: ActionNormalizer | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if num_workers < 0:
            raise ValueError("num_workers must not be negative")
        if metric_space not in {"normalized", "physical"}:
            raise ValueError("metric_space must be 'normalized' or 'physical'")
        if metric_space == "physical" and action_normalizer is None:
            raise ValueError("physical metrics require an ActionNormalizer")
        if len(dataset) == 0:
            raise ValueError("evaluation dataset must not be empty")

        self.model = model
        self.dataset = dataset
        self.device = resolve_device(device)
        self.metric_space = metric_space
        self.action_normalizer = action_normalizer
        self.collator = collator or self._infer_collator(model)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=self.collator,
        )
        self.model.to(self.device)

    @staticmethod
    def _infer_collator(model: BaseVLA) -> VLACollator:
        text_transform = getattr(model, "text_transform", None)
        if not callable(text_transform):
            raise ValueError(
                "provide collator=... or configure the model with a callable text_transform"
            )
        return VLACollator(
            image_transform=getattr(model, "image_transform", None),
            text_transform=text_transform,
        )

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """Return MSE, MAE, and RMSE aggregated across all action dimensions."""
        was_training = self.model.training
        self.model.eval()
        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        value_count = 0
        try:
            for raw_batch in self.loader:
                if not isinstance(raw_batch, VLABatch):
                    raise TypeError("collator must return a VLABatch")
                if raw_batch.actions is None:
                    raise ValueError("evaluation batches must contain target actions")
                batch = raw_batch.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                predictions = self.model(batch)
                targets = batch.actions
                if predictions.shape != targets.shape:
                    raise ValueError(
                        f"prediction shape {tuple(predictions.shape)} does not match "
                        f"target shape {tuple(targets.shape)}"
                    )
                if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
                    raise FloatingPointError("predictions and targets must be finite")
                if self.metric_space == "physical":
                    assert self.action_normalizer is not None
                    predictions = self.action_normalizer.denormalize(predictions)
                    targets = self.action_normalizer.denormalize(targets)
                difference = predictions - targets
                squared_error_sum += float(torch.sum(difference.square()).item())
                absolute_error_sum += float(torch.sum(difference.abs()).item())
                value_count += difference.numel()
        finally:
            self.model.train(was_training)

        if value_count == 0:
            raise ValueError("evaluation loader produced no action values")
        mse = squared_error_sum / value_count
        return {
            "mse": mse,
            "mae": absolute_error_sum / value_count,
            "rmse": mse**0.5,
        }
