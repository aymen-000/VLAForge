"""Action-space specifications."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Describes the dimensions and semantics of a model's action output."""

    dimensions: int
    representation: str = "continuous"
    units: tuple[str, ...] | None = None
    coordinate_frame: str | None = None
    normalized: bool = False

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        if not self.representation.strip():
            raise ValueError("representation must not be empty")
        if self.units is not None and len(self.units) != self.dimensions:
            raise ValueError("units must contain one entry per action dimension")
