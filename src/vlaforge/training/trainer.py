"""Architecture-independent training and validation loops."""

import json
import logging
import random
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset

from vlaforge.core import BaseVLA, VLABatch
from vlaforge.data import ActionNormalizer, VLACollator
from vlaforge.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Train any ``BaseVLA`` model using its own ``compute_loss`` method.

    Pass ``collator`` when the model does not expose ``text_transform`` and
    ``image_transform`` attributes. The trainer never implements
    architecture-specific prediction or loss logic.

    Args:
        model: Model implementing the :class:`BaseVLA` contract.
        train_dataset: Supervised training dataset.
        val_dataset: Optional validation dataset.
        config: Config object, mapping, or YAML path.
        collator: Optional callable converting sample lists to ``VLABatch``.
        resume_from: Optional checkpoint directory to resume from.
        action_normalizer: Optional training-only action statistics to save
            alongside each checkpoint.
    """

    def __init__(
        self,
        model: BaseVLA,
        train_dataset: Dataset[Any],
        val_dataset: Dataset[Any] | None = None,
        config: TrainingConfig | dict[str, Any] | str | Path | None = None,
        collator: Callable[[list[Any]], VLABatch] | None = None,
        resume_from: str | Path | None = None,
        action_normalizer: ActionNormalizer | None = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = TrainingConfig.from_value({} if config is None else config)
        self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)
        self.collator = collator or self._infer_collator(model)
        self.action_normalizer = action_normalizer
        self.output_dir = Path(self.config.output_dir).expanduser()
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        self._seed_everything(self.config.seed)
        self.train_loader = self._make_loader(train_dataset, shuffle=True)
        self.val_loader = (
            self._make_loader(val_dataset, shuffle=False) if val_dataset is not None else None
        )

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("model has no trainable parameters")
        self.optimizer = AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = self._make_scheduler()
        if resume_from is not None:
            self._load_checkpoint(Path(resume_from).expanduser())

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but CUDA is unavailable: {device}")
        return resolved

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _infer_collator(model: BaseVLA) -> VLACollator:
        text_transform = getattr(model, "text_transform", None)
        if not callable(text_transform):
            raise ValueError(
                "provide collator=... or configure the model with a callable text_transform"
            )
        image_transform = getattr(model, "image_transform", None)
        return VLACollator(
            image_transform=image_transform,
            text_transform=text_transform,
        )

    def _make_loader(self, dataset: Dataset[Any], shuffle: bool) -> DataLoader[Any]:
        if len(dataset) == 0:
            raise ValueError("training and validation datasets must not be empty")
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=self.collator,
        )

    def _make_scheduler(self) -> Any:
        if self.config.scheduler == "none":
            return None
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        return StepLR(
            self.optimizer,
            step_size=self.config.step_size,
            gamma=self.config.scheduler_gamma,
        )

    def _run_epoch(self, loader: DataLoader[Any], training: bool) -> float:
        self.model.train(training)
        total_loss = 0.0
        total_examples = 0
        grad_context = torch.enable_grad if training else torch.no_grad
        with grad_context():
            for raw_batch in loader:
                if not isinstance(raw_batch, VLABatch):
                    raise TypeError("collator must return a VLABatch")
                batch = raw_batch.to(self.device, non_blocking=self.device.type == "cuda")
                batch_size = batch.input_ids.shape[0]
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                loss = self.model.compute_loss(batch)
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                    raise ValueError("model.compute_loss(batch) must return a scalar tensor")
                if not torch.isfinite(loss):
                    raise FloatingPointError("model produced a non-finite loss")
                if training:
                    loss.backward()
                    if self.config.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    self.optimizer.step()
                    self.global_step += 1
                total_loss += float(loss.detach().item()) * batch_size
                total_examples += batch_size
        if total_examples == 0:
            raise ValueError("data loader produced no batches")
        return total_loss / total_examples

    def train(self) -> list[dict[str, float]]:
        """Run the configured epochs and return per-epoch loss summaries."""
        history: list[dict[str, float]] = []
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self._run_epoch(self.train_loader, training=True)
            metrics = {"epoch": float(epoch + 1), "train_loss": train_loss}
            val_loader = self.val_loader
            should_validate = (
                val_loader is not None
                and (epoch + 1) % self.config.validation_interval == 0
            )
            improved = False
            if should_validate and val_loader is not None:
                val_loss = self._run_epoch(val_loader, training=False)
                metrics["val_loss"] = val_loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    improved = True
            if self.scheduler is not None:
                self.scheduler.step()
            if improved and self.config.save_best:
                self._save_checkpoint(self.output_dir / "checkpoint_best", epoch + 1)
            self._save_checkpoint(self.output_dir / "checkpoint_last", epoch + 1)
            history.append(metrics)
            logger.info("Epoch %d metrics: %s", epoch + 1, metrics)
        return history

    def _save_checkpoint(self, directory: Path, epoch: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        model_state = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in self.model.state_dict().items()
        }
        save_file(model_state, str(directory / "model.safetensors"))
        state: dict[str, Any] = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(state, directory / "training_state.pt")
        action_spec = asdict(self.model.get_action_spec())
        model_config = getattr(self.model, "config", None)
        metadata = {
            "model_class": type(self.model).__name__,
            "action_spec": action_spec,
            "training_config": self.config.to_dict(),
        }
        if model_config is not None and is_dataclass(model_config):
            serializable_config = asdict(model_config)
            metadata["model_config"] = serializable_config
            model_type = getattr(self.model, "model_type", None)
            if model_type is not None:
                (directory / "config.json").write_text(
                    json.dumps(
                        {"model_type": model_type, "config": serializable_config},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=list), encoding="utf-8"
        )
        if self.action_normalizer is not None:
            self.action_normalizer.save(directory / "normalization.json")

    def _load_checkpoint(self, directory: Path) -> None:
        weights_path = directory / "model.safetensors"
        state_path = directory / "training_state.pt"
        if not weights_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint must contain model.safetensors and training_state.pt: {directory}"
            )
        self.model.load_state_dict(load_file(str(weights_path), device=str(self.device)))
        state = torch.load(state_path, map_location=self.device, weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.start_epoch = int(state["epoch"])
        self.global_step = int(state.get("global_step", 0))
        self.best_val_loss = float(state.get("best_val_loss", float("inf")))
