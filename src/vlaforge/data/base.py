"""Dataset contracts shared by VLAForge data adapters."""

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from vlaforge.core.specs import ActionSpec


class BaseVLADataset(Dataset[dict[str, Any]], ABC):
    """Base class for datasets yielding model-independent VLA samples.

    Each sample must provide ``images`` (camera name to image),
    ``instruction`` (string), and ``action`` (when used for supervised
    training). Dataset adapters own storage details; transforms and collation
    are provided separately by :mod:`vlaforge.data`.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample using the documented sample keys."""

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Return dataset statistics, including training action statistics."""

    @abstractmethod
    def get_action_spec(self) -> ActionSpec:
        """Return the meaning and dimensions of dataset actions."""
