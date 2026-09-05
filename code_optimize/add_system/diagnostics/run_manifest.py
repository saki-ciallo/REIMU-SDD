from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from ..configuration.experiments import ResolvedExperiment
from ..configuration.settings import ExperimentSettings


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def write_run_manifest(
    settings: ExperimentSettings,
    resolved: ResolvedExperiment,
    startup_timings: dict[str, float],
) -> Path:
    """Write reproducibility metadata without collecting credentials or host secrets."""

    manifest = {
        "task_name": settings.task_name,
        "config_sha256": resolved.sha256,
        "config_sources": [str(path) for path in resolved.sources],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": _package_version("transformers"),
            "flash_linear_attention": _package_version("flash-linear-attention"),
            "datasets": _package_version("datasets"),
            "gpu": torch.cuda.get_device_name(),
        },
        "training": {
            "bf16": settings.training.bf16,
            "torch_compile": settings.training.torch_compile,
            "torch_compile_mode": settings.training.torch_compile_mode,
        },
        "startup_seconds": {name: round(seconds, 6) for name, seconds in startup_timings.items()},
    }
    destination = settings.output_dir / "run_manifest.json"
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return destination
