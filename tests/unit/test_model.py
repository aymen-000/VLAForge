"""Tests for the initial VLA action model contract."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from vlaforge import ActionSpec, Observation, VLABatch
from vlaforge.models import VLAActionConfig, VLAActionModel


class TinyVisionEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(-2, -1))


class TinyLanguageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def tiny_tokenizer(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
    del kwargs
    return {
        "input_ids": torch.ones(len(texts), 5, dtype=torch.long),
        "attention_mask": torch.ones(len(texts), 5, dtype=torch.long),
    }


def make_model() -> VLAActionModel:
    config = VLAActionConfig(
        vision_feature_dim=3,
        language_feature_dim=4,
        action_spec=ActionSpec(dimensions=2),
        hidden_dim=8,
        dropout=0.0,
    )
    return VLAActionModel(
        vision_encoder=TinyVisionEncoder(),
        language_encoder=TinyLanguageEncoder(),
        config=config,
        tokenizer=tiny_tokenizer,
    )


def test_model_forward_and_loss_shapes() -> None:
    model = make_model()
    batch = VLABatch(
        images={"front": torch.rand(2, 3, 16, 16)},
        input_ids=torch.ones(2, 5, dtype=torch.long),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
        actions=torch.rand(2, 2),
    )

    predictions = model(batch)
    loss = model.compute_loss(batch)

    assert predictions.shape == (2, 2)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.action_head[0].weight.grad is not None


def test_model_predict_action_uses_injected_tokenizer() -> None:
    model = make_model()
    observation = Observation(
        images={"front": Image.new("RGB", (16, 16))},
        instruction="move the block",
    )

    action = model.predict_action(observation)

    assert action.values.shape == (2,)
    assert action.representation == "continuous"


def test_model_loss_requires_targets() -> None:
    model = make_model()
    batch = VLABatch(
        images={"front": torch.rand(1, 3, 16, 16)},
        input_ids=torch.ones(1, 5, dtype=torch.long),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="batch.actions"):
        model.compute_loss(batch)


def test_model_checkpoint_can_be_reconstructed(tmp_path) -> None:
    model = make_model().eval()
    model.save_pretrained(tmp_path)
    restored = VLAActionModel.from_pretrained(
        tmp_path,
        vision_encoder=TinyVisionEncoder(),
        language_encoder=TinyLanguageEncoder(),
        tokenizer=tiny_tokenizer,
    ).eval()
    batch = VLABatch(
        images={"front": torch.rand(2, 3, 16, 16)},
        input_ids=torch.ones(2, 5, dtype=torch.long),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
    )

    assert torch.allclose(model(batch), restored(batch))
