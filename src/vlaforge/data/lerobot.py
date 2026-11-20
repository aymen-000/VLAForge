"""(Optional) adapter for Hugging Face LeRobot datasets."""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from vlaforge.core.specs import ActionSpec
from vlaforge.data.base import BaseVLADataset


@dataclass(frozen=True, slots=True)
class LeRobotFieldMapping:
    """Map VLAForge sample fields to fields in a LeRobot sample.

    ``camera_fields`` maps VLAForge camera names to LeRobot image keys, for
    example ``{"front": "observation.images.front"}``. When omitted, camera
    keys are inferred from the LeRobot dataset metadata.
    """

    camera_fields: Mapping[str, str] | None = None
    task_field: str = "task_index"
    action_field: str = "action"
    proprioception_field: str | None = None
    episode_field: str = "episode_index"

    def __post_init__(self) -> None:
        if self.camera_fields is not None:
            if not self.camera_fields:
                raise ValueError("camera_fields must not be empty")
            if any(
                not isinstance(name, str)
                or not isinstance(key, str)
                or not name.strip()
                or not key.strip()
                for name, key in self.camera_fields.items()
            ):
                raise ValueError("camera names and field keys must not be empty")
        for name in (self.task_field, self.action_field, self.episode_field):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("task, action, and episode field names must not be empty")
        if self.proprioception_field is not None and (
            not isinstance(self.proprioception_field, str)
            or not self.proprioception_field.strip()
        ):
            raise ValueError("proprioception_field must be None or a non-empty field name")

    @classmethod
    def from_value(
        cls, value: "LeRobotFieldMapping | Mapping[str, Any] | None"
    ) -> "LeRobotFieldMapping":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        allowed = {
            "camera_fields",
            "task_field",
            "action_field",
            "proprioception_field",
            "episode_field",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown LeRobot field mapping keys: {', '.join(unknown)}")
        return cls(**dict(value))


class LeRobotDatasetAdapter(BaseVLADataset):
    """Adapt LeRobot frame samples to VLAForge's dataset contract.

    LeRobot is imported lazily when ``dataset`` is not supplied. Pass an
    explicit :class:`ActionSpec`; LeRobot feature dimensions do not establish
    physical units or coordinate-frame semantics.

    Args:
        action_spec: Required action dimensions and semantics.
        repo_id: LeRobot dataset identifier, such as ``"lerobot/aloha_sim_insertion_human"``.
        root: Optional local dataset directory.
        episodes: Optional LeRobot episode IDs to open.
        field_mapping: Explicit field mapping or mapping dictionary. The
            default ``task_index`` field is resolved through dataset metadata.
        dataset: Existing LeRobot-compatible dataset instance, useful when the
            caller already constructed it or for testing without LeRobot.
        dataset_kwargs: Extra keyword arguments passed to ``LeRobotDataset``.
    """

    def __init__(
        self,
        action_spec: ActionSpec,
        repo_id: str | None = None,
        *,
        root: str | Path | None = None,
        episodes: Sequence[int] | None = None,
        field_mapping: LeRobotFieldMapping | Mapping[str, Any] | None = None,
        dataset: Any | None = None,
        dataset_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(action_spec, ActionSpec):
            raise TypeError("action_spec must be an ActionSpec")
        self._action_spec = action_spec
        self.field_mapping = LeRobotFieldMapping.from_value(field_mapping)

        if dataset is None:
            if not repo_id:
                raise ValueError("provide repo_id or an existing dataset instance")
            if dataset_kwargs and {"repo_id", "root", "episodes"} & set(dataset_kwargs):
                raise ValueError("dataset_kwargs must not override repo_id, root, or episodes")
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ImportError as error:
                raise ImportError(
                    "LeRobotDatasetAdapter requires the optional dependency; "
                    "from the VLAForge checkout install it with "
                    "`python -m pip install -e '.[lerobot]'`."
                ) from error
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=list(episodes) if episodes is not None else None,
                **dict(dataset_kwargs or {}),
            )
        elif any(value is not None for value in (repo_id, root, episodes, dataset_kwargs)):
            raise ValueError(
                "repo_id, root, episodes, and dataset_kwargs cannot be combined with dataset"
            )

        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError("dataset must implement __len__ and __getitem__")
        self.dataset = dataset
        self._camera_fields = self._resolve_camera_fields()
        if not self._camera_fields:
            raise ValueError(
                "could not infer LeRobot camera fields; pass field_mapping.camera_fields"
            )

        features = getattr(dataset, "features", None)
        if isinstance(features, Mapping):
            for field in (*self._camera_fields.values(), self.field_mapping.action_field):
                if field not in features:
                    raise ValueError(f"mapped LeRobot field {field!r} is not in dataset.features")
            action_shape = features[self.field_mapping.action_field].get("shape")
            if action_shape and int(action_shape[-1]) != action_spec.dimensions:
                raise ValueError(
                    "ActionSpec dimensions do not match LeRobot action feature shape: "
                    f"{action_spec.dimensions} != {action_shape[-1]}"
                )

    def _resolve_camera_fields(self) -> dict[str, str]:
        configured = self.field_mapping.camera_fields
        if configured is not None:
            return dict(configured)

        metadata = getattr(self.dataset, "meta", None)
        camera_keys = getattr(metadata, "camera_keys", None)
        if camera_keys is None:
            features = getattr(self.dataset, "features", None)
            if isinstance(features, Mapping):
                camera_keys = [
                    key
                    for key, value in features.items()
                    if isinstance(value, Mapping)
                    and value.get("dtype") in {"image", "video"}
                ]
        if not camera_keys:
            return {}
        result: dict[str, str] = {}
        for key in camera_keys:
            key = str(key)
            camera_name = key.removeprefix("observation.images.")
            result[camera_name or "front"] = key
        return result

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame = self.dataset[index]
        if not isinstance(frame, Mapping):
            raise TypeError("LeRobot dataset items must be mappings")

        missing = [
            field
            for field in (
                *self._camera_fields.values(),
                self.field_mapping.task_field,
                self.field_mapping.action_field,
            )
            if field not in frame
        ]
        if missing:
            raise KeyError(f"LeRobot sample is missing mapped fields: {', '.join(missing)}")

        images = {name: frame[key] for name, key in self._camera_fields.items()}
        instruction = self._instruction(frame[self.field_mapping.task_field])
        action = self._vector(frame[self.field_mapping.action_field], "action")
        if action.numel() != self._action_spec.dimensions:
            raise ValueError(
                f"mapped action has {action.numel()} values; "
                f"ActionSpec requires {self._action_spec.dimensions}"
            )

        sample: dict[str, Any] = {
            "images": images,
            "instruction": instruction,
            "action": action,
        }
        proprio_field = self.field_mapping.proprioception_field
        if proprio_field is not None:
            if proprio_field not in frame:
                raise KeyError(
                    f"LeRobot sample is missing proprioception field {proprio_field!r}"
                )
            sample["proprioception"] = self._vector(frame[proprio_field], "proprioception")
        episode_field = self.field_mapping.episode_field
        if episode_field in frame:
            sample["episode_id"] = self._scalar(frame[episode_field])
        return sample

    @staticmethod
    def _vector(value: Any, label: str) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"LeRobot {label} field must be numeric") from error
        if tensor.ndim != 1 or not torch.isfinite(tensor).all():
            raise ValueError(f"LeRobot {label} field must be a finite vector")
        return tensor

    @staticmethod
    def _scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return value.item()
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass
        return value

    def _instruction(self, value: Any) -> str:
        value = self._scalar(value)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, int):
            tasks = getattr(getattr(self.dataset, "meta", None), "tasks", None)
            if hasattr(tasks, "columns") and "task_index" in tasks.columns:
                matching = tasks.index[tasks["task_index"] == value]
                if len(matching):
                    value = matching[0]
            elif isinstance(tasks, Mapping):
                value = tasks.get(value, tasks.get(str(value), value))
            elif hasattr(tasks, "to_dict"):
                try:
                    rows = tasks.to_dict("records")
                except TypeError:
                    rows = tasks.to_dict()
                if isinstance(rows, list):
                    for row in rows:
                        if row.get("task_index") == value:
                            value = row.get("task", value)
                            break
                elif isinstance(rows, Mapping) and "task_index" in rows:
                    task_indices = rows["task_index"]
                    task_values = rows.get("task", [])
                    for index, task_index in enumerate(task_indices):
                        if task_index == value and index < len(task_values):
                            value = task_values[index]
                            break
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError(
            "task field must contain a non-empty instruction string or a task index "
            "resolvable through dataset.meta.tasks"
        )

    def get_statistics(self, indices: Sequence[int] | None = None) -> dict[str, Any]:
        """Compute action statistics over this dataset or selected frame indices.

        Use ``indices`` for a training subset so validation episodes do not
        contribute to normalization statistics. LeRobot's frame accessor may
        decode images while iterating, depending on its storage backend.
        """
        selected = range(len(self)) if indices is None else indices
        action_dataset = None
        select_columns = getattr(self.dataset, "select_columns", None)
        if callable(select_columns):
            try:
                action_dataset = select_columns(self.field_mapping.action_field)
            except (AttributeError, NotImplementedError):
                # Alternate LeRobot storage readers may not expose an HF column view.
                pass
        count = 0
        mean: torch.Tensor | None = None
        m2: torch.Tensor | None = None
        minimum: torch.Tensor | None = None
        maximum: torch.Tensor | None = None
        for index in selected:
            frame = action_dataset[index] if action_dataset is not None else self.dataset[index]
            action = self._vector(frame[self.field_mapping.action_field], "action")
            if action.numel() != self._action_spec.dimensions:
                raise ValueError("LeRobot action dimension does not match ActionSpec")
            count += 1
            if mean is None:
                mean = action.clone()
                m2 = torch.zeros_like(action)
                minimum = action.clone()
                maximum = action.clone()
            else:
                delta = action - mean
                mean += delta / count
                assert m2 is not None and minimum is not None and maximum is not None
                m2 += delta * (action - mean)
                minimum = torch.minimum(minimum, action)
                maximum = torch.maximum(maximum, action)
        if count == 0 or mean is None or m2 is None or minimum is None or maximum is None:
            raise ValueError("cannot compute statistics for an empty LeRobot selection")
        return {
            "num_samples": count,
            "action_mean": mean,
            "action_std": torch.sqrt(m2 / count),
            "action_min": minimum,
            "action_max": maximum,
        }

    def get_action_spec(self) -> ActionSpec:
        return self._action_spec

    def split_by_episode(
        self,
        validation_fraction: float = 0.2,
        seed: int = 42,
    ) -> tuple[Subset[dict[str, Any]], Subset[dict[str, Any]]]:
        """Return train and validation frame subsets split by whole episodes."""
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        episode_ranges = self._episode_ranges()
        if len(episode_ranges) < 2:
            raise ValueError("at least two episode ranges are required for an episode split")

        episode_ids = list(episode_ranges)
        random.Random(seed).shuffle(episode_ids)
        validation_count = min(
            len(episode_ids) - 1,
            max(1, round(len(episode_ids) * validation_fraction)),
        )
        validation_ids = set(episode_ids[:validation_count])
        train_indices: list[int] = []
        validation_indices: list[int] = []
        for episode_id, (start, stop) in episode_ranges.items():
            destination = validation_indices if episode_id in validation_ids else train_indices
            destination.extend(range(start, stop))
        return Subset(self, train_indices), Subset(self, validation_indices)

    def _episode_ranges(self) -> dict[int, tuple[int, int]]:
        metadata = getattr(getattr(self.dataset, "meta", None), "episodes", None)
        if metadata is None:
            raise ValueError("LeRobot dataset metadata does not expose episode ranges")

        if hasattr(metadata, "to_dict"):
            try:
                rows = metadata.to_dict("records")
            except TypeError:
                rows = metadata.to_dict()
            if isinstance(rows, Mapping):
                rows = [
                    {key: rows[key][position] for key in rows}
                    for position in range(len(rows.get("length", [])))
                ]
        elif isinstance(metadata, Mapping):
            keys = ("episode_index", "length")
            if any(key not in metadata for key in keys):
                raise ValueError("episode metadata must contain episode_index and length")
            rows = [
                {key: metadata[key][position] for key in metadata}
                for position in range(len(metadata["length"]))
            ]
        elif isinstance(metadata, Sequence):
            rows = list(metadata)
        else:
            raise ValueError("unsupported LeRobot episode metadata format")

        rows_by_id: dict[int, Mapping[str, Any]] = {}
        for position, row in enumerate(rows):
            if not isinstance(row, Mapping) or "length" not in row:
                continue
            episode_id = int(row.get("episode_index", position))
            rows_by_id[episode_id] = row

        selected_episodes = getattr(self.dataset, "episodes", None)
        episode_ids = (
            [int(value) for value in selected_episodes]
            if selected_episodes is not None
            else list(rows_by_id)
        )
        ranges: dict[int, tuple[int, int]] = {}
        cursor = 0
        for episode_id in episode_ids:
            row = rows_by_id.get(episode_id)
            if row is None:
                raise ValueError(f"episode {episode_id} is absent from LeRobot metadata")
            length = int(row["length"])
            if length < 1:
                raise ValueError(f"episode {episode_id} has invalid length {length}")
            ranges[episode_id] = (cursor, cursor + length)
            cursor += length
        if cursor != len(self):
            raise ValueError(
                "episode metadata lengths do not match the selected LeRobot dataset length"
            )
        return ranges


__all__ = ["LeRobotDatasetAdapter", "LeRobotFieldMapping"]
