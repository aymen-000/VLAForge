"""Continuous-action prediction metrics."""

import torch


def _validate_pair(predictions: torch.Tensor, targets: torch.Tensor) -> None:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"prediction shape {tuple(predictions.shape)} does not match "
            f"target shape {tuple(targets.shape)}"
        )
    if predictions.numel() == 0:
        raise ValueError("metrics require at least one action value")


def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return mean squared error over all action values."""
    _validate_pair(predictions, targets)
    return torch.mean((predictions - targets) ** 2)


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return mean absolute error over all action values."""
    _validate_pair(predictions, targets)
    return torch.mean(torch.abs(predictions - targets))


def root_mean_squared_error(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Return root mean squared error over all action values."""
    return torch.sqrt(mean_squared_error(predictions, targets))
