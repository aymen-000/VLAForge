"""Small model registry for extensible architecture discovery."""

from collections.abc import Callable, Iterator
from typing import TypeVar

from vlaforge.core.base_model import BaseVLA

ModelType = TypeVar("ModelType", bound=type[BaseVLA])


class ModelRegistry:
    """Maps stable model names to ``BaseVLA`` implementations."""

    def __init__(self) -> None:
        self._models: dict[str, type[BaseVLA]] = {}

    def register(self, name: str) -> Callable[[ModelType], ModelType]:
        """Register a model class under ``name`` and reject duplicate names."""
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("model name must not be empty")

        def decorator(model_class: ModelType) -> ModelType:
            if not issubclass(model_class, BaseVLA):
                raise TypeError("registered models must inherit from BaseVLA")
            if normalized_name in self._models:
                raise ValueError(f"model {normalized_name!r} is already registered")
            self._models[normalized_name] = model_class
            return model_class

        return decorator

    def get(self, name: str) -> type[BaseVLA]:
        """Return a registered model class or raise a descriptive error."""
        try:
            return self._models[name]
        except KeyError as error:
            available = ", ".join(sorted(self._models)) or "none"
            raise KeyError(f"unknown model {name!r}; registered models: {available}") from error

    def names(self) -> tuple[str, ...]:
        """Return registered model names in sorted order."""
        return tuple(sorted(self._models))

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


MODEL_REGISTRY = ModelRegistry()


def register_model(name: str) -> Callable[[ModelType], ModelType]:
    """Register a model in the package-level registry."""
    return MODEL_REGISTRY.register(name)
