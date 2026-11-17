"""YAML and mapping based model construction."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from vlaforge.core import MODEL_REGISTRY, BaseVLA


def _read_model_config(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"model config not found: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(f"invalid model YAML: {path}") from error
    else:
        data = dict(value)

    if not isinstance(data, Mapping):
        raise ValueError("model config must contain a YAML mapping")
    settings = data.get("model", data)
    if not isinstance(settings, Mapping):
        raise ValueError("'model' config section must be a mapping")
    result = dict(settings)
    name = result.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model config must specify a non-empty 'name'")
    result["name"] = name.strip()
    return result


def build_model(
    config: Mapping[str, Any] | str | Path,
    **components: Any,
) -> BaseVLA:
    """Construct a registered model from a mapping or YAML file.

    Architecture classes implement ``from_config(config, **components)``.
    External modules such as pretrained encoders and tokenizers are passed as
    keyword arguments so the base package does not download or select them.
    """
    settings = _read_model_config(config)
    model_name = settings.pop("name")
    try:
        model_class = MODEL_REGISTRY.get(model_name)
    except KeyError as error:
        raise ValueError(str(error)) from error

    constructor = getattr(model_class, "from_config", None)
    if not callable(constructor):
        raise TypeError(
            f"registered model {model_name!r} must implement from_config(config, **components)"
        )
    return constructor(settings, **components)


__all__ = ["build_model"]
