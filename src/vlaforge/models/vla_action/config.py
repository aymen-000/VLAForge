"""Configuration for the VLM plus action-head architecture."""

from dataclasses import dataclass

from vlaforge.core.specs import ActionSpec


@dataclass(frozen=True, slots=True)
class VLAActionConfig:
    """Shape and behavior settings for :class:`VLAActionModel`."""

    vision_feature_dim: int
    language_feature_dim: int
    action_spec: ActionSpec
    hidden_dim: int = 512
    dropout: float = 0.1
    camera_name: str = "front"
    freeze_encoders: bool = True

    def __post_init__(self) -> None:
        if self.vision_feature_dim < 1 or self.language_feature_dim < 1:
            raise ValueError("encoder feature dimensions must be positive")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if not self.camera_name.strip():
            raise ValueError("camera_name must not be empty")
