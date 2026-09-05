from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

type ConfigValue = object
type ConfigMapping = dict[str, ConfigValue]


class StrictSafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: StrictSafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> ConfigMapping:
    loader.flatten_mapping(node)
    result: ConfigMapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise TypeError(
                f"YAML mapping keys must be strings, got {type(key).__name__} "
                f"at line {key_node.start_mark.line + 1}."
            )
        if key in result:
            raise ValueError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}.")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_mapping(path: str | Path) -> ConfigMapping:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        loaded = yaml.load(handle, Loader=StrictSafeLoader)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Top-level YAML value in {resolved} must be a mapping, got {type(loaded).__name__}."
        )
    return loaded


def deep_merge(
    base: Mapping[str, ConfigValue],
    overlay: Mapping[str, ConfigValue],
) -> ConfigMapping:
    """Recursively merge mappings while replacing scalar/list leaves."""

    merged: ConfigMapping = deepcopy(dict(base))
    for key, overlay_value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(overlay_value, Mapping):
            base_model_type = base_value.get("model_type")
            overlay_model_type = overlay_value.get("model_type")
            if (
                base_model_type is not None
                and overlay_model_type is not None
                and base_model_type != overlay_model_type
            ):
                merged[key] = deepcopy(dict(overlay_value))
            else:
                merged[key] = deep_merge(base_value, overlay_value)
        else:
            merged[key] = deepcopy(overlay_value)
    return merged


def _canonical_digest(values: Mapping[str, ConfigValue]) -> str:
    payload = json.dumps(
        values,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolvedExperiment:
    values: ConfigMapping
    sources: tuple[Path, ...]
    sha256: str

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.values,
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
        return destination

    def build_model_config(self):
        """Construct the model config from the resolved ``model`` section."""

        from .factory import build_add_config

        model_values = self.values.get("model")
        if not isinstance(model_values, Mapping):
            raise TypeError("Resolved experiment must contain a model mapping.")
        return build_add_config(model_values)

    def build_settings(self):
        """Build validated run, data, model, loss, and training settings."""

        from .settings import build_experiment_settings

        return build_experiment_settings(self)


def resolve_experiment(
    paths: Sequence[str | Path],
    *,
    overrides: Mapping[str, ConfigValue] | None = None,
) -> ResolvedExperiment:
    if not paths:
        raise ValueError("At least one configuration path is required.")

    resolved_values: ConfigMapping = {}
    resolved_paths = tuple(Path(path).expanduser().resolve() for path in paths)
    for path in resolved_paths:
        resolved_values = deep_merge(resolved_values, load_yaml_mapping(path))
    if overrides:
        resolved_values = deep_merge(resolved_values, overrides)

    return ResolvedExperiment(
        values=resolved_values,
        sources=resolved_paths,
        sha256=_canonical_digest(resolved_values),
    )
