"""Continuous action normalization utilities."""

import json
from pathlib import Path

import torch


class ActionNormalizer:
    """Normalize and denormalize actions using training-set mean and std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, epsilon: float = 1e-6) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        std_tensor = torch.as_tensor(std, dtype=torch.float32)
        if mean_tensor.ndim != 1 or std_tensor.shape != mean_tensor.shape:
            raise ValueError("mean and std must be vectors with identical shapes")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
            raise ValueError("mean and std must contain finite values")
        if torch.any(std_tensor < 0):
            raise ValueError("std values must not be negative")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.mean = mean_tensor
        self.std = std_tensor.clamp_min(epsilon)

    @classmethod
    def from_actions(cls, actions: torch.Tensor, epsilon: float = 1e-6) -> "ActionNormalizer":
        """Fit statistics from training actions shaped ``(N, D)``."""
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError("actions must have non-empty shape (num_samples, dimensions)")
        if not torch.isfinite(actions).all():
            raise ValueError("actions must contain finite values")
        return cls(
            mean=actions.mean(dim=0),
            std=actions.std(dim=0, unbiased=False),
            epsilon=epsilon,
        )

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Map actions to standardized values."""
        return (actions - self.mean.to(actions.device)) / self.std.to(actions.device)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Map standardized predictions back to the original action units."""
        return actions * self.std.to(actions.device) + self.mean.to(actions.device)

    def to_dict(self) -> dict[str, list[float]]:
        """Return JSON-compatible normalization statistics."""
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, values: dict[str, list[float]]) -> "ActionNormalizer":
        """Rebuild an action normalizer from serialized statistics."""
        try:
            return cls(
                mean=torch.tensor(values["mean"], dtype=torch.float32),
                std=torch.tensor(values["std"], dtype=torch.float32),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("normalization data must contain mean and std vectors") from error

    def save(self, path: str | Path) -> None:
        """Write statistics to a JSON file."""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ActionNormalizer":
        """Read statistics from a JSON file."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"normalization file not found: {source}")
        try:
            values = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid normalization JSON: {source}") from error
        if not isinstance(values, dict):
            raise ValueError("normalization JSON must contain an object")
        return cls.from_dict(values)
