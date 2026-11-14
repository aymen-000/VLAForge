"""Image conversion, resizing, and normalization."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


class ImageTransform:
    """Convert an image to normalized RGB float tensor with shape ``(3, H, W)``.

    Args:
        size: Optional output size as ``(height, width)``.
        mean: Per-channel normalization mean, in RGB order.
        std: Per-channel normalization standard deviation, in RGB order.
    """

    def __init__(
        self,
        size: tuple[int, int] | None = (224, 224),
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        if size is not None and (len(size) != 2 or min(size) < 1):
            raise ValueError("size must contain positive (height, width) values")
        if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
            raise ValueError("mean and std must each contain three values; std must be positive")
        self.size = size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Any) -> torch.Tensor:
        """Return an RGB image tensor in normalized ``(C, H, W)`` format."""
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            tensor = torch.from_numpy(array)
        elif isinstance(image, torch.Tensor):
            tensor = image.detach()
        else:
            tensor = torch.as_tensor(np.asarray(image))

        if tensor.ndim != 3:
            raise ValueError("image must have three dimensions")
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
        elif tensor.shape[0] in (1, 3, 4):
            tensor = tensor[:3]
        else:
            raise ValueError("image must be channel-last or channel-first RGB/RGBA")
        if tensor.shape[0] == 1:
            tensor = tensor.expand(3, -1, -1)

        tensor = tensor.to(dtype=torch.float32)
        if tensor.numel() and tensor.max() > 1:
            tensor = tensor / 255.0
        if self.size is not None and tensor.shape[-2:] != self.size:
            tensor = functional.interpolate(
                tensor.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
            ).squeeze(0)
        return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)
