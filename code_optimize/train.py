from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

CODE_OPTIMIZE_ROOT = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the optimized ADD model.")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="YAML file. Repeat in common-to-experiment merge order.",
    )
    parser.add_argument("--gpu-id", type=int)
    return parser.parse_args()


def configure_visible_gpu(paths: list[str], override: int | None) -> int | None:
    gpu_id = override
    if gpu_id is None:
        for path in paths:
            with Path(path).open(encoding="utf-8") as handle:
                values = yaml.safe_load(handle) or {}
            run_values = values.get("run", {})
            if isinstance(run_values, dict) and "gpu_id" in run_values:
                gpu_id = run_values["gpu_id"]
    if gpu_id is not None:
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError("gpu_id must be a non-negative integer or null.")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return gpu_id


def main() -> None:
    arguments = parse_arguments()
    arguments.config = [str(Path(path).expanduser().resolve()) for path in arguments.config]
    # Keep all relative optimized-package paths rooted at code_optimize. The
    # repository-level configs and utilities belong to the legacy boundary.
    os.chdir(CODE_OPTIMIZE_ROOT)
    configure_visible_gpu(arguments.config, arguments.gpu_id)

    from add_system.configuration import resolve_experiment
    from add_system.training.run import configure_logging, run_training

    configure_logging()
    resolved = resolve_experiment(arguments.config)
    run_training(resolved)


if __name__ == "__main__":
    main()
