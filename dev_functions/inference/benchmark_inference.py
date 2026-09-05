from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Allow this standalone benchmark to import the repository runtime.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.modeling_add import ADDModel
from src.registration import register_add_for_auto
from utilis.model_test_utils import (
    checkpoint_step,
    find_last_checkpoint,
    has_saved_model,
    resolve_model_path,
)

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_NAME = "inference_efficiency.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Batch-1 latency and real-time factor for a saved ADDModel."
        )
    )
    parser.add_argument(
        "--model_path",
        "--model-dir",
        dest="model_path",
        required=True,
        help=(
            "Checkpoint path, run name under outputs, or run directory. "
            "A run directory automatically selects its largest checkpoint-*."
        ),
    )
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--audio_seconds", type=float, default=4.0)
    parser.add_argument("--warmup_iterations", type=int, default=20)
    parser.add_argument("--benchmark_iterations", type=int, default=100)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="CUDA autocast precision used during the benchmark.",
    )
    parser.add_argument(
        "--output_name",
        default=DEFAULT_OUTPUT_NAME,
        help="JSON filename written inside the resolved checkpoint directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero.")
    if args.audio_seconds <= 0:
        raise ValueError("audio_seconds must be greater than zero.")
    if args.warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative.")
    if args.benchmark_iterations <= 0:
        raise ValueError("benchmark_iterations must be greater than zero.")
    if Path(args.output_name).name != args.output_name:
        raise ValueError("output_name must be a filename, not a path.")


def resolve_checkpoint(model_path: str | Path) -> Path:
    path = resolve_model_path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"model_path does not exist or is not a directory: {path}"
        )
    if checkpoint_step(path) is not None:
        if not has_saved_model(path):
            raise FileNotFoundError(
                f"Checkpoint does not contain a loadable ADDModel: {path}"
            )
        return path

    checkpoint = find_last_checkpoint(path)
    if checkpoint is not None and has_saved_model(checkpoint):
        logger.info("Selected latest checkpoint: %s", checkpoint)
        return checkpoint
    if has_saved_model(path):
        return path
    raise FileNotFoundError(
        f"No loadable ADDModel found under {path}. Expected config.json and "
        "model.safetensors/pytorch_model.bin, either directly or in checkpoint-*."
    )


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def restore_configured_trainability(model: ADDModel) -> None:
    """Reapply SSL freezing because requires_grad is not stored in checkpoints."""
    configure_trainability = getattr(
        model.frontend,
        "_configure_ssl_trainability",
        None,
    )
    if callable(configure_trainability):
        configure_trainability()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty sequence.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def forward_once(
    model: ADDModel,
    input_values: torch.Tensor,
    autocast_context: Any,
) -> torch.Tensor:
    with autocast_context:
        outputs = model(
            input_values=input_values,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    if outputs.logits is None:
        raise RuntimeError("Model did not return logits during benchmarking.")
    return outputs.logits


def benchmark(
    model: ADDModel,
    input_values: torch.Tensor,
    *,
    audio_seconds: float,
    warmup_iterations: int,
    benchmark_iterations: int,
    dtype: str,
) -> dict[str, float | int | str]:
    autocast_enabled = dtype == "bf16"

    def make_autocast_context() -> Any:
        if not autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        logits = None
        for _ in range(warmup_iterations):
            logits = forward_once(model, input_values, make_autocast_context())
        torch.cuda.synchronize()

        latencies_seconds = []
        for _ in range(benchmark_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = forward_once(model, input_values, make_autocast_context())
            torch.cuda.synchronize()
            latencies_seconds.append(time.perf_counter() - start)

    if logits is None or not torch.isfinite(logits).all():
        raise FloatingPointError("Model produced non-finite logits during benchmarking.")

    mean_seconds = statistics.fmean(latencies_seconds)
    median_seconds = percentile(latencies_seconds, 0.50)
    p95_seconds = percentile(latencies_seconds, 0.95)
    rtf = mean_seconds / audio_seconds
    return {
        "batch_size": 1,
        "warmup_iterations": warmup_iterations,
        "benchmark_iterations": benchmark_iterations,
        "latency_mean_ms": mean_seconds * 1_000.0,
        "latency_p50_ms": median_seconds * 1_000.0,
        "latency_p95_ms": p95_seconds * 1_000.0,
        "rtf": rtf,
        "realtime_factor_x": 1.0 / rtf,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "benchmark_inference.py requires CUDA because ADD/FLA uses GPU kernels."
        )
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16.")

    register_add_for_auto()
    checkpoint = resolve_checkpoint(args.model_path)
    num_samples = round(args.sample_rate * args.audio_seconds)
    if num_samples <= 0:
        raise ValueError("sample_rate * audio_seconds must produce at least one sample.")
    torch.manual_seed(0)
    input_values = torch.randn(
        1,
        num_samples,
        device="cuda",
        dtype=torch.float32,
    )

    logger.info("Loading checkpoint: %s", checkpoint)
    model = ADDModel.from_pretrained(str(checkpoint)).cuda().eval()
    restore_configured_trainability(model)
    counts = parameter_counts(model)
    logger.info(
        "Parameters: total=%s trainable=%s frozen=%s",
        f"{counts['total']:,}",
        f"{counts['trainable']:,}",
        f"{counts['frozen']:,}",
    )
    logger.info(
        "Benchmarking Batch-1 input: %s samples (%.3f seconds), dtype=%s",
        num_samples,
        args.audio_seconds,
        args.dtype,
    )

    timing = benchmark(
        model,
        input_values,
        audio_seconds=args.audio_seconds,
        warmup_iterations=args.warmup_iterations,
        benchmark_iterations=args.benchmark_iterations,
        dtype=args.dtype,
    )
    result = {
        "model": {
            "checkpoint": str(checkpoint),
            "architecture_type": getattr(model.config, "architecture_type", None),
            "model_architecture": getattr(model.config, "model_architecture", None),
            "parameters": counts,
        },
        "input": {
            "sample_rate": args.sample_rate,
            "audio_seconds": args.audio_seconds,
            "num_samples": num_samples,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda_device_index": torch.cuda.current_device(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "dtype": args.dtype,
            "torch_compile": False,
        },
        "metrics": timing,
    }
    output_path = checkpoint / args.output_name
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    logger.info(
        "Latency: mean=%.3f ms, p50=%.3f ms, p95=%.3f ms",
        timing["latency_mean_ms"],
        timing["latency_p50_ms"],
        timing["latency_p95_ms"],
    )
    logger.info(
        "RTF=%.6f (%.2fx real-time)",
        timing["rtf"],
        timing["realtime_factor_x"],
    )
    logger.info("Saved benchmark result: %s", output_path)


if __name__ == "__main__":
    main()
