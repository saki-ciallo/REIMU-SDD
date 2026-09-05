from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from rawboost_opt import Algorithm, RawBoost, RawBoostConfig
from rawboost_opt.parallel import ParallelBatchProcessor, ParallelConfig
from rawboost_opt.torch_backend import augment_torch


def _measure(function, repeats: int, warmup: int = 1) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        elapsed.append(time.perf_counter() - started)
    return statistics.median(elapsed), statistics.mean(elapsed)


def _result(
    implementation: str,
    resources: int,
    batch_size: int,
    median: float,
    mean: float,
) -> dict[str, object]:
    return {
        "implementation": implementation,
        "resources": resources,
        "batch_size": batch_size,
        "median_ms_per_batch": median * 1_000,
        "mean_ms_per_batch": mean * 1_000,
        "median_ms_per_sample": median * 1_000 / batch_size,
        "samples_per_second": batch_size / median,
    }


def benchmark_numpy_processes(
    waveforms: np.ndarray,
    seeds: list[int],
    config: RawBoostConfig,
    resources: list[int],
    repeats: int,
) -> list[dict[str, object]]:
    rows = []
    for workers in resources:
        parallel = (
            ParallelConfig(kind="serial", workers=1)
            if workers == 1
            else ParallelConfig(kind="process", workers=workers)
        )
        with ParallelBatchProcessor(
            config,
            Algorithm.LNL_ISD_SSI,
            parallel,
        ) as processor:
            median, mean = _measure(
                lambda: processor.process(waveforms, seeds=seeds),
                repeats,
            )
        rows.append(_result("numpy_process", workers, len(waveforms), median, mean))
    return rows


def benchmark_torch_cpu(
    waveforms: np.ndarray,
    seeds: list[int],
    config: RawBoostConfig,
    resources: list[int],
    repeats: int,
) -> list[dict[str, object]]:
    source = torch.from_numpy(waveforms)
    augmenter = RawBoost(config)
    parameters = [
        augmenter.parameters(waveforms.shape[1], Algorithm.LNL_ISD_SSI, seed=seed) for seed in seeds
    ]
    rows = []
    for threads in resources:
        torch.set_num_threads(threads)
        median, mean = _measure(
            lambda: augmenter.augment_batch(
                source,
                Algorithm.LNL_ISD_SSI,
                seeds=seeds,
                backend="torch",
            ),
            repeats,
        )
        rows.append(_result("torch_cpu_end_to_end", threads, len(source), median, mean))

        median, mean = _measure(
            lambda: augment_torch(
                source,
                Algorithm.LNL_ISD_SSI,
                parameters,
                config,
            ),
            repeats,
        )
        rows.append(_result("torch_cpu_execution_only", threads, len(source), median, mean))
    return rows


def save_report(
    output_dir: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cpu_scaling.json").write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "cpu_scaling.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# RawBoost CPU Scaling",
        "",
        "Algorithm 4 on four-second, 16 kHz float32 waveforms. Resources means NumPy worker "
        "processes or Torch intra-op threads.",
        "",
        "| implementation | resources | batch | ms/sample | samples/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {implementation} | {resources} | {batch_size} | "
            "{median_ms_per_sample:.3f} | {samples_per_second:.1f} |".format(**row)
        )
    (output_dir / "cpu_scaling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RawBoost algorithm 4 CPU scaling benchmark.")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "results")
    parser.add_argument("--resources", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available_cpus = len(os.sched_getaffinity(0))
    resources = sorted(set(args.resources))
    invalid = [value for value in resources if value <= 0 or value > available_cpus]
    if invalid:
        raise ValueError(f"Resources must lie in [1, {available_cpus}], got {invalid}.")

    config = RawBoostConfig()
    num_samples = int(config.sampling_rate * args.seconds)
    waveforms = (
        np.random.default_rng(2026)
        .normal(
            0.0,
            0.1,
            (args.batch_size, num_samples),
        )
        .astype(np.float32)
    )
    seeds = [config.seed + index for index in range(args.batch_size)]
    rows = benchmark_numpy_processes(waveforms, seeds, config, resources, args.repeats)
    rows.extend(benchmark_torch_cpu(waveforms, seeds, config, resources, args.repeats))
    metadata = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "available_cpus": available_cpus,
        "physical_cpu_note": "Ryzen 9 9950X: 16 physical cores, 32 logical CPUs",
        "algorithm": Algorithm.LNL_ISD_SSI.value,
        "batch_size": args.batch_size,
        "num_samples": num_samples,
        "repeats": args.repeats,
        "resources": resources,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "config": {
            **asdict(config),
            "fir_backend": config.fir_backend.value,
        },
    }
    save_report(args.output_dir, rows, metadata)
    print((args.output_dir / "cpu_scaling.md").resolve())


if __name__ == "__main__":
    main()
