from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from .config_utils import merge_yaml_configs


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _parse_startup_config(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, dict[str, object]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config")
    parser.add_argument("--experiment-config")
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--torch_compile", nargs="?", const=True, type=_parse_bool)
    startup_args, _ = parser.parse_known_args(argv)

    config_paths = [
        path
        for path in (startup_args.config, startup_args.experiment_config)
        if path is not None
    ]
    merged_config = merge_yaml_configs(*config_paths) if config_paths else {}
    return startup_args, merged_config


def configure_visible_gpu_from_argv(
    argv: Sequence[str] | None = None,
) -> int | None:
    """Set CUDA visibility from merged YAML or --gpu_id before torch is imported."""

    startup_args, merged_config = _parse_startup_config(argv)
    gpu_id = (
        startup_args.gpu_id
        if startup_args.gpu_id is not None
        else merged_config.get("gpu_id")
    )
    if gpu_id is None:
        return None
    if isinstance(gpu_id, bool) or not isinstance(gpu_id, int):
        raise TypeError(f"gpu_id must be a non-negative integer, got {gpu_id!r}.")
    if gpu_id < 0:
        raise ValueError(f"gpu_id must be non-negative, got {gpu_id}.")

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return gpu_id


def configure_torch_compile_stance_from_argv(
    argv: Sequence[str] | None = None,
) -> str:
    """Apply force-eager before importing modules with torch.compile decorators."""

    startup_args, merged_config = _parse_startup_config(argv)
    torch_compile = (
        startup_args.torch_compile
        if startup_args.torch_compile is not None
        else merged_config.get("torch_compile", False)
    )
    if not isinstance(torch_compile, bool):
        raise TypeError(f"torch_compile must be a boolean, got {torch_compile!r}.")
    if torch_compile:
        return "default"

    import torch

    torch.compiler.set_stance("force_eager")
    return "force_eager"


__all__ = [
    "configure_torch_compile_stance_from_argv",
    "configure_visible_gpu_from_argv",
]
