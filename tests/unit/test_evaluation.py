"""Tests for action metrics and offline evaluation."""

import torch
from torch import nn
from torch.utils.data import Dataset

from vlaforge import Action, ActionSpec, BaseVLA, Observation, VLABatch
from vlaforge.data import ActionNormalizer
from vlaforge.evaluation import Evaluator, mean_absolute_error, mean_squared_error


class TargetDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"target": torch.tensor([float(index + 1)])}


class ZeroActionModel(BaseVLA):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, batch: VLABatch) -> torch.Tensor:
        return torch.zeros_like(batch.actions) + self.weight

    def predict_action(self, observation: Observation) -> Action:
        del observation
        return Action(torch.zeros(1))

    def compute_loss(self, batch: VLABatch) -> torch.Tensor:
        if batch.actions is None:
            raise ValueError("actions required")
        return self(batch).square().mean()

    def get_action_spec(self) -> ActionSpec:
        return ActionSpec(dimensions=1)


def target_collator(samples: list[dict[str, torch.Tensor]]) -> VLABatch:
    actions = torch.stack([sample["target"] for sample in samples])
    return VLABatch(
        images={"front": torch.zeros(len(samples), 3, 1, 1)},
        input_ids=torch.ones(len(samples), 1, dtype=torch.long),
        attention_mask=torch.ones(len(samples), 1, dtype=torch.long),
        actions=actions,
    )


def test_action_metrics() -> None:
    predictions = torch.tensor([[1.0, 2.0]])
    targets = torch.tensor([[2.0, 4.0]])

    assert torch.isclose(mean_squared_error(predictions, targets), torch.tensor(2.5))
    assert torch.isclose(mean_absolute_error(predictions, targets), torch.tensor(1.5))


def test_evaluator_reports_aggregated_metrics() -> None:
    evaluator = Evaluator(
        model=ZeroActionModel(),
        dataset=TargetDataset(),
        collator=target_collator,
        batch_size=2,
        device="cpu",
    )

    results = evaluator.evaluate()

    assert results["mse"] == 2.5
    assert results["mae"] == 1.5
    assert results["rmse"] == 2.5**0.5


def test_evaluator_converts_to_physical_units() -> None:
    evaluator = Evaluator(
        model=ZeroActionModel(),
        dataset=TargetDataset(),
        collator=target_collator,
        batch_size=2,
        device="cpu",
        metric_space="physical",
        action_normalizer=ActionNormalizer(
            mean=torch.tensor([10.0]), std=torch.tensor([2.0])
        ),
    )

    results = evaluator.evaluate()

    assert results["mse"] == 10.0
    assert results["mae"] == 3.0
