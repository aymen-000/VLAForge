"""Opt-in integration check against a real LeRobot dataset."""

import os
from pathlib import Path

import pytest
import torch

from vlaforge.core import ActionSpec
from vlaforge.data import LeRobotDatasetAdapter, LeRobotFieldMapping


REPO_ID = os.environ.get("VLAFORGE_LEROBOT_REPO_ID")
pytestmark = pytest.mark.skipif(
    not REPO_ID,
    reason="set VLAFORGE_LEROBOT_REPO_ID to enable the real LeRobot integration test",
)


def test_real_lerobot_dataset_adapter() -> None:
    lerobot = pytest.importorskip(
        "lerobot.datasets.lerobot_dataset",
        reason="install the optional VLAForge LeRobot extra to run this integration test",
    )
    repo_id = REPO_ID
    assert repo_id is not None
    root_value = os.environ.get("VLAFORGE_LEROBOT_ROOT")
    dataset_root = Path(root_value).expanduser() if root_value else None
    print(
        f"Opening LeRobot dataset {repo_id!r} with episodes [0, 1]. "
        f"Root: {dataset_root if dataset_root is not None else 'LeRobot default cache'}. "
        "LeRobot/Hugging Face progress bars below show any files being downloaded.",
        flush=True,
    )
    dataset = lerobot.LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=[0, 1],
    )
    print(
        f"LeRobot dataset opened: {len(dataset)} frames across "
        f"{dataset.num_episodes} episodes.",
        flush=True,
    )
    if dataset.num_episodes < 2:
        pytest.skip("the configured LeRobot dataset must contain at least two episodes")

    action_field = os.environ.get("VLAFORGE_LEROBOT_ACTION_FIELD", "action")
    task_field = os.environ.get("VLAFORGE_LEROBOT_TASK_FIELD", "task_index")
    proprioception_field = os.environ.get("VLAFORGE_LEROBOT_PROPRIOCEPTION_FIELD")
    action_shape = dataset.features[action_field].get("shape")
    if not action_shape or len(action_shape) != 1:
        pytest.fail(f"expected a single-step vector action feature, got {action_shape!r}")

    adapter = LeRobotDatasetAdapter(
        action_spec=ActionSpec(dimensions=int(action_shape[0])),
        dataset=dataset,
        field_mapping=LeRobotFieldMapping(
            task_field=task_field,
            action_field=action_field,
            proprioception_field=proprioception_field,
        ),
    )

    sample = adapter[0]
    assert sample["images"]
    assert isinstance(sample["instruction"], str) and sample["instruction"].strip()
    assert sample["action"].shape == (int(action_shape[0]),)
    assert torch.isfinite(sample["action"]).all()
    assert "episode_id" in sample

    stats = adapter.get_statistics(indices=[0])
    assert stats["num_samples"] == 1
    assert torch.isfinite(stats["action_mean"]).all()

    train_data, validation_data = adapter.split_by_episode(
        validation_fraction=0.5,
        seed=19,
    )
    assert train_data.indices and validation_data.indices
    assert set(train_data.indices).isdisjoint(validation_data.indices)

