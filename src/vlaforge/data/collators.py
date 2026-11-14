"""Batch collation for dataset samples."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from vlaforge.core.batch import VLABatch
from vlaforge.data.transforms.images import ImageTransform
from vlaforge.data.transforms.language import TokenizerTransform


class VLACollator:
    """Transform samples and combine them into a :class:`VLABatch`.

    Provide the tokenizer used by the selected language model. It is injected
    instead of importing a specific Transformers tokenizer in the data module.
    """

    def __init__(
        self,
        tokenizer: Callable[[Sequence[str]], dict[str, Any]] | None = None,
        image_transform: Callable[[Any], torch.Tensor] | None = None,
        text_transform: Callable[[Sequence[str]], Mapping[str, Any]] | None = None,
        action_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.image_transform = image_transform or ImageTransform()
        if text_transform is None:
            if tokenizer is None:
                raise ValueError("provide either a tokenizer or a text_transform")
            text_transform = TokenizerTransform(tokenizer)
        self.text_transform = text_transform
        self.action_transform = action_transform

    def __call__(self, samples: Sequence[dict[str, Any]]) -> VLABatch:
        if not samples:
            raise ValueError("cannot collate an empty batch")

        camera_names = tuple(samples[0]["images"])
        expected_cameras = set(camera_names)
        if not camera_names:
            raise ValueError("each sample must contain at least one image")
        image_batches: dict[str, torch.Tensor] = {}
        for camera in camera_names:
            if any(set(sample["images"]) != expected_cameras for sample in samples):
                raise ValueError("all samples in a batch must have identical camera names")
            transformed = [
                self.image_transform(sample["images"][camera]) for sample in samples
            ]
            if any(tensor.shape != transformed[0].shape for tensor in transformed[1:]):
                raise ValueError(
                    f"transformed images for camera {camera!r} have inconsistent shapes"
                )
            image_batches[camera] = torch.stack(transformed)

        instructions = [sample["instruction"] for sample in samples]
        tokens = self.text_transform(instructions)
        input_ids = self._as_tensor(tokens, "input_ids", torch.long)
        attention_mask = self._as_tensor(tokens, "attention_mask", torch.long)

        actions = self._optional_stack(samples, "action")
        if actions is not None and self.action_transform is not None:
            actions = self.action_transform(actions)
        proprioception = self._optional_stack(samples, "proprioception")
        metadata = {
            key: [sample[key] for sample in samples]
            for key in ("episode_id",)
            if any(key in sample for sample in samples)
        }
        return VLABatch(
            images=image_batches,
            input_ids=input_ids,
            attention_mask=attention_mask,
            actions=actions,
            proprioception=proprioception,
            metadata=metadata or None,
        )

    @staticmethod
    def _as_tensor(
        tokens: Mapping[str, Any], key: str, dtype: torch.dtype
    ) -> torch.Tensor:
        if key not in tokens:
            raise ValueError(f"tokenizer output must contain {key!r}")
        tensor = torch.as_tensor(tokens[key], dtype=dtype)
        if tensor.ndim != 2:
            raise ValueError(f"tokenizer field {key!r} must have shape (batch, sequence)")
        return tensor

    @staticmethod
    def _optional_stack(samples: Sequence[dict[str, Any]], key: str) -> torch.Tensor | None:
        present = [key in sample and sample[key] is not None for sample in samples]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"either all samples or no samples must provide {key!r}")
        values = [torch.as_tensor(sample[key], dtype=torch.float32) for sample in samples]
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError(f"all {key!r} values in a batch must have the same shape")
        return torch.stack(values)
