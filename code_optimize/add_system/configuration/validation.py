from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import PretrainedConfig


def positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}.")
    return value


def positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}.")
    resolved = float(value)
    if resolved <= 0.0:
        raise ValueError(f"{field_name} must be positive, got {value}.")
    return resolved


def probability(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}.")
    resolved = float(value)
    if not 0.0 <= resolved < 1.0:
        raise ValueError(f"{field_name} must satisfy 0 <= value < 1, got {value}.")
    return resolved


def boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}.")
    return value


def optional_positive_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    return positive_int(value, field_name=field_name)


def one_of(value: object, choices: set[str], *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}.")
    resolved = value.strip().lower()
    if resolved not in choices:
        supported = ", ".join(sorted(choices))
        raise ValueError(f"{field_name} must be one of {supported}, got {value!r}.")
    return resolved


def coerce_config[T: PretrainedConfig](
    value: T | Mapping[str, Any] | None,
    config_type: type[T],
    *,
    field_name: str,
    default: bool = False,
) -> T | None:
    """Restore a nested HF config while rejecting a mismatched model type."""

    if value is None:
        return config_type() if default else None
    if isinstance(value, config_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{field_name} must be {config_type.__name__}, a mapping, or None; "
            f"got {type(value).__name__}."
        )

    values = dict(value)
    model_type = values.pop("model_type", None)
    compatible_types = {
        config_type.model_type,
        *getattr(config_type, "legacy_model_types", ()),
    }
    if model_type is not None and model_type not in compatible_types:
        supported = ", ".join(sorted(compatible_types))
        raise TypeError(f"{field_name}.model_type must be one of {supported}, got {model_type!r}.")
    return config_type(**values)
