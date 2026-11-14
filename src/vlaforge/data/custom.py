"""JSON Lines based dataset adapter for user-managed image datasets."""

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from vlaforge.core.specs import ActionSpec
from vlaforge.data.base import BaseVLADataset


class CustomVLADataset(BaseVLADataset):
    """Load samples from a JSON Lines manifest and image files.

    ``root`` may be a directory containing ``manifest.jsonl`` or the manifest
    file itself. Each line contains ``instruction``, ``action``, and either
    ``image`` (one relative image path) or ``images`` (a camera-to-path map).
    Optional fields are ``proprioception`` and ``episode_id``. Image paths are
    resolved relative to the manifest's parent directory.

    Args:
        root: Dataset directory or path to its JSON Lines manifest.
        action_spec: Required semantics for the action vectors.

    Raises:
        FileNotFoundError: If the manifest or a referenced image is missing.
        ValueError: If a manifest line is malformed or lacks required fields.
    """

    def __init__(self, root: str | Path, action_spec: ActionSpec) -> None:
        path = Path(root).expanduser()
        manifest = path / "manifest.jsonl" if path.is_dir() else path
        if not manifest.is_file():
            raise FileNotFoundError(f"dataset manifest not found: {manifest}")

        self.root = manifest.parent
        self.manifest_path = manifest
        self._action_spec = action_spec
        self._records: list[dict[str, Any]] = []
        self._actions: list[torch.Tensor] = []

        with manifest.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON on manifest line {line_number}: {error.msg}"
                    ) from error
                normalized = self._validate_record(record, line_number)
                self._records.append(normalized)
                self._actions.append(torch.tensor(normalized["action"], dtype=torch.float32))

        if not self._records:
            raise ValueError(f"dataset manifest contains no samples: {manifest}")

    def _validate_record(self, record: Any, line_number: int) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} must contain a JSON object")
        instruction = record.get("instruction")
        action = record.get("action")
        images = record.get("images")
        if images is None and "image" in record:
            images = {"front": record["image"]}
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"manifest line {line_number} needs a non-empty instruction")
        if not isinstance(images, dict) or not images:
            raise ValueError(f"manifest line {line_number} needs image or images paths")
        if not all(isinstance(camera, str) and isinstance(value, str) for camera, value in images.items()):
            raise ValueError(f"manifest line {line_number} image paths must be strings")
        if not isinstance(action, list) or len(action) != self._action_spec.dimensions:
            raise ValueError(
                f"manifest line {line_number} action must have "
                f"{self._action_spec.dimensions} values"
            )
        try:
            action_tensor = torch.tensor(action, dtype=torch.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"manifest line {line_number} action must be numeric") from error
        if not torch.isfinite(action_tensor).all():
            raise ValueError(f"manifest line {line_number} action values must be finite")

        resolved_images: dict[str, str] = {}
        for camera, image_path in images.items():
            resolved = (self.root / image_path).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"image for camera {camera!r} on line {line_number} not found: {resolved}"
                )
            resolved_images[camera] = str(resolved)

        normalized = dict(record)
        normalized["images"] = resolved_images
        normalized["action"] = action_tensor.tolist()
        normalized["instruction"] = instruction
        proprioception = normalized.get("proprioception")
        if proprioception is not None:
            try:
                proprio_tensor = torch.tensor(proprioception, dtype=torch.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"manifest line {line_number} proprioception must be numeric"
                ) from error
            if proprio_tensor.ndim != 1 or not torch.isfinite(proprio_tensor).all():
                raise ValueError(
                    f"manifest line {line_number} proprioception must be a finite vector"
                )
            normalized["proprioception"] = proprio_tensor.tolist()
        return normalized

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        images: dict[str, Image.Image] = {}
        for camera, image_path in record["images"].items():
            try:
                with Image.open(image_path) as image:
                    images[camera] = image.convert("RGB").copy()
            except (UnidentifiedImageError, OSError) as error:
                raise ValueError(f"could not read image file: {image_path}") from error

        sample: dict[str, Any] = {
            "images": images,
            "instruction": record["instruction"],
            "action": torch.tensor(record["action"], dtype=torch.float32),
        }
        if "proprioception" in record:
            sample["proprioception"] = torch.tensor(
                record["proprioception"], dtype=torch.float32
            )
        if "episode_id" in record:
            sample["episode_id"] = record["episode_id"]
        return sample

    def get_statistics(self) -> dict[str, Any]:
        """Compute action mean and population standard deviation.

        Call this on the training dataset only; validation and test data must
        not contribute to normalization statistics.
        """
        actions = torch.stack(self._actions)
        return {
            "num_samples": len(self),
            "action_mean": actions.mean(dim=0),
            "action_std": actions.std(dim=0, unbiased=False),
            "action_min": actions.amin(dim=0),
            "action_max": actions.amax(dim=0),
        }

    def get_action_spec(self) -> ActionSpec:
        return self._action_spec
