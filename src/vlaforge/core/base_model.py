"""Abstract model interface shared by VLA architectures."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from vlaforge.core.action import Action
from vlaforge.core.batch import VLABatch
from vlaforge.core.observation import Observation
from vlaforge.core.specs import ActionSpec


class BaseVLA(nn.Module, ABC):
    """Base contract for VLA models accepted by framework components."""

    @abstractmethod
    def forward(self, batch: VLABatch) -> torch.Tensor:
        """Return predicted actions for a collated batch."""

    @abstractmethod
    def predict_action(self, observation: Observation) -> Action:
        """Predict an action for one unbatched observation."""

    @abstractmethod
    def compute_loss(self, batch: VLABatch) -> torch.Tensor:
        """Compute a scalar training loss from a batch with target actions."""

    @abstractmethod
    def get_action_spec(self) -> ActionSpec:
        """Return the action-space specification for this model."""
