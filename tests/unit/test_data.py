"""Tests for data adapters, transforms, and collation."""

import builtins
import json
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest
import torch

from vlaforge.core import ActionSpec
from vlaforge.data import (
    ActionNormalizer,
    CustomVLADataset,
    ImageTransform,
    LeRobotDatasetAdapter,
    LeRobotFieldMapping,
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


class ActionOnlyView:
    def __init__(self, frames: list[dict[str, Any]], field: str) -> None:
        self.frames = frames
        self.field = field

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {self.field: self.frames[index][self.field]}


class FakeLeRobotDataset:
    def __init__(self) -> None:
        episode_lengths = (2, 1, 3)
        self.frames = [
            self._frame(index, episode)
            for episode, length in enumerate(episode_lengths)
            for index in range(sum(episode_lengths[:episode]), sum(episode_lengths[:episode]) + length)
        ]
        self.meta = SimpleNamespace(
            camera_keys=["observation.images.front"],
            tasks={0: "move the block", 1: "open the drawer", 2: "close the drawer"},
            episodes={"episode_index": [0, 1, 2], "length": list(episode_lengths)},
        )
        self.episodes = None
        self.features = {
            "observation.images.front": {"dtype": "video", "shape": [8, 8, 3]},
            "action": {"dtype": "float32", "shape": [2]},
            "observation.state": {"dtype": "float32", "shape": [3]},
        }
        self.decoded_frame_reads = 0

    @staticmethod
    def _frame(index: int, episode: int) -> dict[str, Any]:
        return {
            "observation.images.front": torch.full((3, 8, 8), index, dtype=torch.uint8),
            "task_index": torch.tensor(episode),
            "action": torch.tensor([float(index), float(index + 1)]),
            "observation.state": torch.tensor([float(index), 0.0, 1.0]),
            "episode_index": torch.tensor(episode),
        }

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.decoded_frame_reads += 1
        return self.frames[index]

    def select_columns(self, field: str) -> ActionOnlyView:
        return ActionOnlyView(self.frames, field)


def make_lerobot_adapter() -> tuple[LeRobotDatasetAdapter, FakeLeRobotDataset]:
    source = FakeLeRobotDataset()
    adapter = LeRobotDatasetAdapter(
        action_spec=ActionSpec(dimensions=2),
        dataset=source,
        field_mapping=LeRobotFieldMapping(
            proprioception_field="observation.state",
        ),
    )
    return adapter, source


def test_lerobot_adapter_maps_sample_fields() -> None:
    dataset, _ = make_lerobot_adapter()

    sample = dataset[0]

    assert set(sample["images"]) == {"front"}
    assert sample["instruction"] == "move the block"
    assert torch.equal(sample["action"], torch.tensor([0.0, 1.0]))
    assert torch.equal(sample["proprioception"], torch.tensor([0.0, 0.0, 1.0]))
    assert sample["episode_id"] == 0
    assert dataset.get_action_spec() == ActionSpec(dimensions=2)


def test_lerobot_adapter_statistics_support_selected_indices() -> None:
    dataset, source = make_lerobot_adapter()

    statistics = dataset.get_statistics(indices=[0, 1])

    assert statistics["num_samples"] == 2
    assert torch.allclose(statistics["action_mean"], torch.tensor([0.5, 1.5]))
    assert torch.allclose(statistics["action_std"], torch.tensor([0.5, 0.5]))
    assert torch.equal(statistics["action_min"], torch.tensor([0.0, 1.0]))
    assert torch.equal(statistics["action_max"], torch.tensor([1.0, 2.0]))
    assert source.decoded_frame_reads == 0


def test_lerobot_adapter_splits_whole_episodes_deterministically() -> None:
    dataset, _ = make_lerobot_adapter()

    train_data, validation_data = dataset.split_by_episode(
        validation_fraction=0.34,
        seed=11,
    )

    train_episodes = {dataset[index]["episode_id"] for index in train_data.indices}
    validation_episodes = {
        dataset[index]["episode_id"] for index in validation_data.indices
    }
    assert train_episodes.isdisjoint(validation_episodes)
    assert train_data.indices and validation_data.indices
    assert sorted(train_data.indices + validation_data.indices) == list(range(len(dataset)))


def test_lerobot_adapter_checks_action_spec_dimensions() -> None:
    with pytest.raises(ValueError, match="ActionSpec dimensions"):
        LeRobotDatasetAdapter(
            action_spec=ActionSpec(dimensions=3),
            dataset=FakeLeRobotDataset(),
        )


def test_lerobot_adapter_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_lerobot(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "lerobot.datasets.lerobot_dataset":
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_lerobot)
    with pytest.raises(ImportError, match=r"vlaforge\[lerobot\]"):
        LeRobotDatasetAdapter(
            action_spec=ActionSpec(dimensions=2),
            repo_id="example/robot_dataset",
            field_mapping={"camera_fields": {"front": "observation.images.front"}},
        )
