"""Tests for the public core contracts."""

import pytest
import torch

from vlaforge import Action, ActionSpec, BaseVLA, Observation, VLABatch
from vlaforge.core.registry import ModelRegistry


def test_action_spec_validates_units() -> None:
    spec = ActionSpec(dimensions=2, units=("meter", "radian"))

    assert spec.dimensions == 2

    with pytest.raises(ValueError, match="one entry per action dimension"):
        ActionSpec(dimensions=2, units=("meter",))


def test_observation_requires_an_image_and_instruction() -> None:
    observation = Observation(images={"front": object()}, instruction="move forward")

    assert observation.instruction == "move forward"
    with pytest.raises(ValueError, match="at least one camera"):
        Observation(images={}, instruction="move forward")


def test_action_requires_positive_horizon() -> None:
    action = Action(values=torch.zeros(7), horizon=1)

    assert action.values.shape == (7,)
    with pytest.raises(ValueError, match="at least 1"):
        Action(values=torch.zeros(7), horizon=0)


def test_batch_checks_text_tensor_shapes() -> None:
    batch = VLABatch(
        images={"front": torch.zeros(2, 3, 16, 16)},
        input_ids=torch.ones(2, 5, dtype=torch.long),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
        actions=torch.zeros(2, 7),
    )

    assert batch.actions is not None
    with pytest.raises(ValueError, match="same shape"):
        VLABatch(
            images={"front": torch.zeros(2, 3, 16, 16)},
            input_ids=torch.ones(2, 5, dtype=torch.long),
            attention_mask=torch.ones(2, 4, dtype=torch.long),
        )


def test_registry_rejects_duplicate_names() -> None:
    registry = ModelRegistry()

    class ExampleModel(BaseVLA):
        def forward(self, batch: VLABatch) -> torch.Tensor:
            return torch.zeros(1)

        def predict_action(self, observation: Observation) -> Action:
            return Action(torch.zeros(1))

        def compute_loss(self, batch: VLABatch) -> torch.Tensor:
            return torch.zeros(())

        def get_action_spec(self) -> ActionSpec:
            return ActionSpec(dimensions=1)

    registry.register("example")(ExampleModel)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("example")(ExampleModel)


def test_registry_reports_unknown_names() -> None:
    registry = ModelRegistry()

    with pytest.raises(KeyError, match="unknown model"):
        registry.get("missing")
