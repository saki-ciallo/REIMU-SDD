from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ..models import ADDModel
from .model_report import parameter_counts


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CUDA Batch-1 latency and real-time factor for an ADD checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--audio-seconds", type=_positive_float, default=4.0)
    parser.add_argument("--sampling-rate", type=_positive_int, default=16_000)
    parser.add_argument("--warmup-steps", type=_positive_int, default=10)
    parser.add_argument("--measurement-steps", type=_positive_int, default=50)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile-mode",
        choices=["default", "max-autotune-no-cudagraphs"],
        default="default",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def benchmark(
    checkpoint: Path,
    *,
    audio_seconds: float,
    sampling_rate: int,
    warmup_steps: int,
    measurement_steps: int,
    compile_model: bool,
    compile_mode: str,
) -> dict[str, int | float | str | bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("ADD benchmarking requires CUDA.")
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    model = ADDModel.from_pretrained(checkpoint).cuda().eval()
    if compile_model:
        model = torch.compile(model, mode=compile_mode)
    num_samples = round(audio_seconds * sampling_rate)
    input_values = torch.randn(1, num_samples, device="cuda", dtype=torch.float32)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup_steps):
            output = model(input_values=input_values, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        if not torch.isfinite(output.logits).all():
            raise FloatingPointError("Warmup produced non-finite logits.")

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(measurement_steps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(measurement_steps)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            output = model(input_values=input_values, use_cache=False, return_dict=True)
            end.record()
        torch.cuda.synchronize()

    latencies_ms = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    median_ms = statistics.median(latencies_ms)
    total_parameters, trainable_parameters = parameter_counts(model)
    return {
        "checkpoint": str(checkpoint),
        "device": torch.cuda.get_device_name(),
        "audio_seconds": audio_seconds,
        "sampling_rate": sampling_rate,
        "num_samples": num_samples,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "compiled": compile_model,
        "compile_mode": compile_mode,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "batch_1_latency_mean_ms": statistics.fmean(latencies_ms),
        "batch_1_latency_median_ms": median_ms,
        "batch_1_latency_p95_ms": _percentile(latencies_ms, 0.95),
        "real_time_factor": median_ms / 1000.0 / audio_seconds,
        "real_time_speedup": audio_seconds / (median_ms / 1000.0),
    }


def main() -> None:
    arguments = parse_arguments()
    results = benchmark(
        arguments.checkpoint,
        audio_seconds=arguments.audio_seconds,
        sampling_rate=arguments.sampling_rate,
        warmup_steps=arguments.warmup_steps,
        measurement_steps=arguments.measurement_steps,
        compile_model=arguments.compile,
        compile_mode=arguments.compile_mode,
    )
    destination = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else arguments.checkpoint.expanduser().resolve() / "inference_benchmark.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=True))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
