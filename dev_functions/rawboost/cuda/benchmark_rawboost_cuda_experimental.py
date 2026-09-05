from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    # Allow this standalone experiment to import the repository runtime.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from datasets import load_from_disk

from utilis.RawBoost import process_Rawboost_feature
from dev_functions.rawboost.cuda.rawboost_cuda_experimental import (
    ExperimentalCudaRawBoostAlgo4,
    apply_rawboost_algo4_cuda,
    apply_rawboost_algo4_numpy,
    sample_rawboost_algo4_parameters,
)
from utilis.rawboost_utils import RawBoostArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original NumPy RawBoost algo4 with experimental CUDA FFT."
    )
    parser.add_argument(
        "--dataset-path",
        default="../datasets/ASVspoof2019_16k_4s_fixed",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "relative_rmse": rmse / max(reference_rms, np.finfo(np.float64).eps),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The experimental RawBoost benchmark requires CUDA.")
    if args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("batch-size and repeats must be positive.")

    dataset = load_from_disk(args.dataset_path)[args.split]
    waveforms = np.stack(
        [
            np.asarray(dataset[index]["input_values"], dtype=np.float32)
            for index in range(args.batch_size)
        ]
    )
    rawboost_config = RawBoostArguments(rawboost_algo=4)
    cuda_rawboost = ExperimentalCudaRawBoostAlgo4(rawboost_config).cuda()
    process_args = cuda_rawboost.process_args
    print("RawBoost algo4 defaults:", vars(process_args))

    original_outputs = []
    replay_outputs = []
    parameter_sets = []
    for index, waveform in enumerate(waveforms):
        sample_seed = args.seed + index
        np.random.seed(sample_seed)
        original_outputs.append(
            process_Rawboost_feature(
                waveform.copy(),
                16_000,
                process_args,
                4,
            )
        )
        np.random.seed(sample_seed)
        parameters = sample_rawboost_algo4_parameters(
            waveform.shape[-1],
            process_args,
            sampling_rate=16_000,
        )
        parameter_sets.append(parameters)
        replay_outputs.append(
            apply_rawboost_algo4_numpy(waveform, parameters)
        )

    original = np.stack(original_outputs).astype(np.float32)
    replay = np.stack(replay_outputs).astype(np.float32)
    cuda_inputs = torch.from_numpy(waveforms).cuda()
    torch.cuda.synchronize()
    cuda_outputs = apply_rawboost_algo4_cuda(cuda_inputs, parameter_sets)
    torch.cuda.synchronize()
    cuda_numpy = cuda_outputs.cpu().numpy()

    print("NumPy parameter replay:", error_metrics(original, replay))
    print("CUDA FFT vs original:", error_metrics(original, cuda_numpy))
    print(
        "Output:",
        {
            "shape": tuple(cuda_outputs.shape),
            "dtype": str(cuda_outputs.dtype),
            "finite": bool(torch.isfinite(cuda_outputs).all()),
        },
    )

    cpu_times = []
    cpu_replay_times = []
    sampling_times = []
    cuda_times = []
    peak_extra_memory = 0
    for repeat in range(args.repeats):
        start = time.perf_counter()
        for index, waveform in enumerate(waveforms):
            np.random.seed(args.seed + repeat * args.batch_size + index)
            process_Rawboost_feature(
                waveform.copy(),
                16_000,
                process_args,
                4,
            )
        cpu_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        for waveform, parameters in zip(waveforms, parameter_sets):
            apply_rawboost_algo4_numpy(waveform, parameters)
        cpu_replay_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        for index, waveform in enumerate(waveforms):
            np.random.seed(args.seed + repeat * args.batch_size + index)
            sample_rawboost_algo4_parameters(
                waveform.shape[-1],
                process_args,
                sampling_rate=16_000,
            )
        sampling_times.append(time.perf_counter() - start)

        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        start = time.perf_counter()
        apply_rawboost_algo4_cuda(cuda_inputs, parameter_sets)
        torch.cuda.synchronize()
        cuda_times.append(time.perf_counter() - start)
        peak_extra_memory = max(
            peak_extra_memory,
            torch.cuda.max_memory_allocated() - baseline_memory,
        )

    mean_cpu = float(np.mean(cpu_times))
    mean_cpu_replay = float(np.mean(cpu_replay_times))
    mean_sampling = float(np.mean(sampling_times))
    mean_cuda = float(np.mean(cuda_times))
    print(
        "Execution time:",
        {
            "cpu_original_total_ms": 1_000 * mean_cpu,
            "cpu_same_parameters_ms": 1_000 * mean_cpu_replay,
            "parameter_sampling_ms": 1_000 * mean_sampling,
            "cuda_fft_ms": 1_000 * mean_cuda,
            "cuda_serial_total_ms": 1_000 * (mean_sampling + mean_cuda),
            "same_parameter_execution_speedup": mean_cpu_replay / mean_cuda,
            "serial_end_to_end_speedup": mean_cpu / (mean_sampling + mean_cuda),
            "cuda_peak_extra_mib": peak_extra_memory / 2**20,
        },
    )


if __name__ == "__main__":
    main()
