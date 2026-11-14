"""Tests for data adapters, transforms, and collation."""

import json

from PIL import Image
import pytest
import torch

from vlaforge.core import ActionSpec
from vlaforge.data import (
    ActionNormalizer,
    CustomVLADataset,
    ImageTransform,
    VLACollator,
)


def test_action_normalizer_round_trips() -> None:
    actions = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
    normalizer = ActionNormalizer.from_actions(actions)

    normalized = normalizer.normalize(actions)

    assert torch.allclose(normalizer.denormalize(normalized), actions)


def test_action_normalizer_handles_constant_dimensions() -> None:
    normalizer = ActionNormalizer.from_actions(torch.tensor([[2.0], [2.0]]))

    assert torch.isfinite(normalizer.normalize(torch.tensor([[2.0]]))).all()


def test_action_normalizer_saves_and_loads(tmp_path) -> None:
    normalizer = ActionNormalizer(
        mean=torch.tensor([1.0, 2.0]), std=torch.tensor([3.0, 4.0])
    )
    path = tmp_path / "normalization.json"

    normalizer.save(path)
    restored = ActionNormalizer.load(path)

    assert torch.equal(restored.mean, normalizer.mean)
    assert torch.equal(restored.std, normalizer.std)


def test_image_transform_converts_and_resizes() -> None:
    transform = ImageTransform(size=(8, 10))
    result = transform(Image.new("RGB", (20, 12), color=(255, 0, 0)))

    assert result.shape == (3, 8, 10)
    assert torch.isfinite(result).all()


def test_collator_builds_vla_batch() -> None:
    def tokenizer(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        return {
            "input_ids": torch.ones(len(texts), 4, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
        }

    collator = VLACollator(tokenizer, image_transform=ImageTransform(size=(8, 8)))
    sample = {
        "images": {"front": Image.new("RGB", (8, 8))},
        "instruction": "move forward",
        "action": torch.zeros(7),
    }

    batch = collator([sample, sample])

    assert batch.images["front"].shape == (2, 3, 8, 8)
    assert batch.input_ids.shape == (2, 4)
    assert batch.actions is not None and batch.actions.shape == (2, 7)


def test_custom_dataset_reads_manifest_and_computes_stats(tmp_path) -> None:
    (tmp_path / "images").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "images" / "frame.png")
    record = {
        "image": "images/frame.png",
        "instruction": "move forward",
        "action": [0.0, 1.0],
        "episode_id": "ep-1",
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = CustomVLADataset(tmp_path, ActionSpec(dimensions=2))
    sample = dataset[0]

    assert sample["images"]["front"].size == (4, 4)
    assert sample["episode_id"] == "ep-1"
    assert torch.equal(dataset.get_statistics()["action_mean"], torch.tensor([0.0, 1.0]))


def test_custom_dataset_rejects_wrong_action_dimensions(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"image": "frame.png", "instruction": "go", "action": [1.0]}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="2 values"):
        CustomVLADataset(tmp_path, ActionSpec(dimensions=2))
