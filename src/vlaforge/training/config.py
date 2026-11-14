"""Validated training configuration and YAML loading."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Settings for the generic single-device PyTorch trainer."""

    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    step_size: int = 5
    scheduler_gamma: float = 0.1
    validation_interval: int = 1
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    output_dir: str = "checkpoints"
    save_best: bool = True
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "optimizer", self.optimizer.lower())
        object.__setattr__(self, "scheduler", self.scheduler.lower())
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if self.optimizer.lower() != "adamw":
            raise ValueError("optimizer must be 'adamw'")
        if self.scheduler.lower() not in {"none", "cosine", "step"}:
            raise ValueError("scheduler must be one of: none, cosine, step")
        if self.step_size < 1:
            raise ValueError("step_size must be at least 1")
        if self.scheduler_gamma <= 0:
            raise ValueError("scheduler_gamma must be positive")
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must not be negative")
        if self.device != "auto" and self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device string")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when set")

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingConfig":
        """Read configuration from a YAML file.

        The YAML root may contain a ``training`` mapping or contain the
        settings directly.
        """
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"training config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict):
            raise ValueError("training config must contain a YAML mapping")
        settings = data.get("training", data)
        if not isinstance(settings, dict):
            raise ValueError("'training' config section must be a YAML mapping")
        checkpoint = settings.pop("checkpoint", None)
        if isinstance(checkpoint, dict):
            settings.setdefault("output_dir", checkpoint.get("output_dir", "checkpoints"))
            settings.setdefault("save_best", checkpoint.get("save_best", True))
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise ValueError(f"unknown training config fields: {', '.join(unknown)}")
        try:
            return cls(**settings)
        except TypeError as error:
            raise ValueError(f"invalid training config: {error}") from error

    @classmethod
    def from_value(cls, value: "TrainingConfig | dict[str, Any] | str | Path") -> "TrainingConfig":
        """Construct a config from an instance, mapping, or YAML path."""
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            return cls.from_file(value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("config must be TrainingConfig, a mapping, or a YAML path")

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""
        return asdict(self)
