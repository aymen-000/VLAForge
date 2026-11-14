"""Tests for the generic training engine."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from vlaforge import Action, ActionSpec, BaseVLA, Observation, VLABatch
from vlaforge.training import Trainer, TrainingConfig


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = torch.tensor([float(index), float(index + 1)])
        return {"input": value, "action": value[:1] * 0.5}


class TinyModel(BaseVLA):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2, 1)

    def forward(self, batch: VLABatch) -> torch.Tensor:
        return self.head(batch.input_ids.float())

    def predict_action(self, observation: Observation) -> Action:
        del observation
        return Action(torch.zeros(1))

    def compute_loss(self, batch: VLABatch) -> torch.Tensor:
        if batch.actions is None:
            raise ValueError("target actions are required")
        return nn.functional.mse_loss(self(batch), batch.actions)

    def get_action_spec(self) -> ActionSpec:
        return ActionSpec(dimensions=1)


def tiny_collator(samples: list[dict[str, torch.Tensor]]) -> VLABatch:
    return VLABatch(
        images={"front": torch.zeros(len(samples), 3, 1, 1)},
        input_ids=torch.stack([sample["input"] for sample in samples]),
        attention_mask=torch.ones(len(samples), 2, dtype=torch.long),
        actions=torch.stack([sample["action"] for sample in samples]),
    )


def test_training_saves_last_and_best_checkpoints(tmp_path: Path) -> None:
    config = TrainingConfig(
        epochs=2,
        batch_size=2,
        scheduler="none",
        device="cpu",
        output_dir=str(tmp_path),
    )
    trainer = Trainer(
        model=TinyModel(),
        train_dataset=TinyDataset(),
        val_dataset=TinyDataset(),
        config=config,
        collator=tiny_collator,
    )

    history = trainer.train()

    assert len(history) == 2
    assert "train_loss" in history[0] and "val_loss" in history[0]
    assert (tmp_path / "checkpoint_last" / "model.safetensors").is_file()
    assert (tmp_path / "checkpoint_best" / "training_state.pt").is_file()


def test_training_config_rejects_unknown_scheduler() -> None:
    with pytest.raises(ValueError, match="scheduler"):
        TrainingConfig(scheduler="linear")
