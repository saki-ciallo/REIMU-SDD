from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for import_root in (REPOSITORY_ROOT, EXPERIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from utilis.RawBoost import process_Rawboost_feature  # noqa: E402

from rawboost_opt import Algorithm, FIRBackend, RawBoost, RawBoostConfig  # noqa: E402
from rawboost_opt.numpy_backend import augment_numpy  # noqa: E402
from rawboost_opt.parallel import (  # noqa: E402
    ParallelBatchProcessor,
    ParallelConfig,
)
from rawboost_opt.parameters import sample_parameters  # noqa: E402


def _legacy_arguments(config: RawBoostConfig) -> SimpleNamespace:
    return SimpleNamespace(
        N_f=config.nonlinearity_order,
        nBands=config.num_bands,
        minF=config.min_frequency,
        maxF=config.max_frequency,
        minBW=config.min_bandwidth,
        maxBW=config.max_bandwidth,
        minCoeff=config.min_coefficients,
        maxCoeff=config.max_coefficients,
        minG=config.min_gain,
        maxG=config.max_gain,
        minBiasLinNonLin=config.min_nonlinear_bias,
        maxBiasLinNonLin=config.max_nonlinear_bias,
        P=config.impulse_percent,
        g_sd=config.impulse_gain,
        SNRmin=config.min_snr,
        SNRmax=config.max_snr,
    )


def _measure(function, repeats: int, warmup: int = 1) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), statistics.mean(samples)


def _record(
    rows: list[dict[str, object]],
    *,
    section: str,
    implementation: str,
    algorithm: int,
    batch_size: int,
    median_seconds: float,
    mean_seconds: float,
    workers: int = 1,
) -> None:
    rows.append(
        {
            "section": section,
            "implementation": implementation,
            "algorithm": algorithm,
            "batch_size": batch_size,
            "workers": workers,
            "median_ms_per_batch": median_seconds * 1_000,
            "mean_ms_per_batch": mean_seconds * 1_000,
            "median_ms_per_sample": median_seconds * 1_000 / batch_size,
            "samples_per_second": batch_size / median_seconds,
        }
    )


def benchmark_algorithms(
    waveform: np.ndarray,
    config: RawBoostConfig,
    repeats: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    legacy_args = _legacy_arguments(config)
    overlap_augmenter = RawBoost(config)
    direct_config = replace(config, fir_backend=FIRBackend.DIRECT)
    direct_augmenter = RawBoost(direct_config)

    for algorithm in Algorithm:
        np.random.seed(config.seed)
        median, mean = _measure(
            lambda algorithm=algorithm: process_Rawboost_feature(
                waveform,
                config.sampling_rate,
                legacy_args,
                algorithm.value,
            ),
            repeats,
        )
        _record(
            rows,
            section="algorithms",
            implementation="legacy_numpy_direct",
            algorithm=algorithm.value,
            batch_size=1,
            median_seconds=median,
            mean_seconds=mean,
        )

        seed = config.seed + algorithm.value
        median, mean = _measure(
            lambda algorithm=algorithm, seed=seed: overlap_augmenter.augment(
                waveform,
                algorithm,
                seed=seed,
                backend="numpy",
            ),
            repeats,
        )
        _record(
            rows,
            section="algorithms",
            implementation="optimized_numpy_overlap_add",
            algorithm=algorithm.value,
            batch_size=1,
            median_seconds=median,
            mean_seconds=mean,
        )

        median, mean = _measure(
            lambda algorithm=algorithm, seed=seed: direct_augmenter.augment(
                waveform,
                algorithm,
                seed=seed,
                backend="numpy",
            ),
            repeats,
        )
        _record(
            rows,
            section="algorithms",
            implementation="optimized_numpy_direct",
            algorithm=algorithm.value,
            batch_size=1,
            median_seconds=median,
            mean_seconds=mean,
        )
    return rows


def benchmark_algo4_components(
    waveform: np.ndarray,
    config: RawBoostConfig,
    repeats: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    algorithm = Algorithm.LNL_ISD_SSI

    def sample() -> object:
        return sample_parameters(
            len(waveform),
            algorithm,
            config,
            np.random.default_rng(config.seed),
        )

    parameters = sample()
    median, mean = _measure(sample, repeats)
    _record(
        rows,
        section="algo4_components",
        implementation="parameter_sampling",
        algorithm=algorithm.value,
        batch_size=1,
        median_seconds=median,
        mean_seconds=mean,
    )
    median, mean = _measure(
        lambda: augment_numpy(waveform, algorithm, parameters, config),
        repeats,
    )
    _record(
        rows,
        section="algo4_components",
        implementation="waveform_application",
        algorithm=algorithm.value,
        batch_size=1,
        median_seconds=median,
        mean_seconds=mean,
    )
    return rows


def benchmark_parallel(
    waveforms: np.ndarray,
    config: RawBoostConfig,
    repeats: int,
    worker_counts: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seeds = [config.seed + index for index in range(len(waveforms))]
    choices = [("serial", 1)] + [
        (kind, workers)
        for kind in ("thread", "process")
        for workers in worker_counts
        if workers > 1
    ]
    for kind, workers in choices:
        with ParallelBatchProcessor(
            config,
            Algorithm.LNL_ISD_SSI,
            ParallelConfig(kind=kind, workers=workers),
        ) as processor:
            median, mean = _measure(
                lambda: processor.process(waveforms, seeds=seeds),
                repeats,
            )
        _record(
            rows,
            section="algo4_parallel",
            implementation=kind,
            algorithm=Algorithm.LNL_ISD_SSI.value,
            batch_size=len(waveforms),
            workers=workers,
            median_seconds=median,
            mean_seconds=mean,
        )
    return rows


def benchmark_cuda(
    waveforms: np.ndarray,
    config: RawBoostConfig,
    repeats: int,
) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        return []
    augmenter = RawBoost(config)
    source = torch.from_numpy(waveforms).cuda()
    seeds = [config.seed + index for index in range(len(waveforms))]
    rows = []
    for algorithm in Algorithm:

        def run(algorithm: Algorithm = algorithm) -> None:
            augmenter.augment_batch(
                source,
                algorithm,
                seeds=seeds,
                backend="torch",
            )
            torch.cuda.synchronize()

        median, mean = _measure(run, repeats, warmup=2)
        _record(
            rows,
            section="cuda_batch",
            implementation="optimized_torch_fft",
            algorithm=algorithm.value,
            batch_size=len(waveforms),
            median_seconds=median,
            mean_seconds=mean,
        )
    return rows


def _markdown_table(rows: list[dict[str, object]], section: str) -> str:
    selected = [row for row in rows if row["section"] == section]
    lines = [
        "| implementation | algo | batch | workers | ms/sample | samples/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| {implementation} | {algorithm} | {batch_size} | {workers} | "
            "{median_ms_per_sample:.3f} | {samples_per_second:.1f} |".format(**row)
        )
    return "\n".join(lines)


def save_results(
    output_dir: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# RawBoost Benchmark",
        "",
        "All values use 16 kHz, four-second float32 waveforms. Lower ms/sample is better.",
        "",
    ]
    for section, title in (
        ("algorithms", "Algorithm 0-8, Single Sample CPU"),
        ("algo4_components", "Algorithm 4 Components"),
        ("algo4_parallel", "Algorithm 4 CPU Parallelism"),
        ("cuda_batch", "Algorithm 0-8, CUDA Batch"),
    ):
        if any(row["section"] == section for row in rows):
            report.extend([f"## {title}", "", _markdown_table(rows, section), ""])
    (output_dir / "benchmark.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark legacy and optimized RawBoost.")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "results")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--skip-parallel", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RawBoostConfig()
    num_samples = int(config.sampling_rate * args.seconds)
    rng = np.random.default_rng(2026)
    waveforms = rng.normal(0.0, 0.1, (args.batch_size, num_samples)).astype(np.float32)
    rows = benchmark_algorithms(waveforms[0], config, args.repeats)
    rows.extend(benchmark_algo4_components(waveforms[0], config, args.repeats))
    if not args.skip_parallel:
        rows.extend(benchmark_parallel(waveforms, config, args.repeats, args.workers))
    if not args.skip_cuda:
        rows.extend(benchmark_cuda(waveforms, config, args.repeats))
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "num_samples": num_samples,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "config": {
            **asdict(config),
            "fir_backend": config.fir_backend.value,
        },
    }
    save_results(args.output_dir, rows, metadata)
    print((args.output_dir / "benchmark.md").resolve())


if __name__ == "__main__":
    main()
