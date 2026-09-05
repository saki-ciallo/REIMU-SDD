from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


def load_yaml_config(path: str | Path) -> dict[str, object]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing YAML config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"YAML config must contain a mapping at the top level: {path}")
    return dict(loaded)


def merge_yaml_configs(*paths: str | Path) -> dict[str, object]:
    merged: dict[str, object] = {}
    for path in paths:
        merged.update(load_yaml_config(path))
    return merged


def save_yaml_config(config: Mapping[str, object], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return path


__all__ = ["load_yaml_config", "merge_yaml_configs", "save_yaml_config"]
