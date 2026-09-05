from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from rawboost_opt.config import Algorithm, FIRBackend, RawBoostConfig
from rawboost_opt.huggingface import ArrowMapConfig, build_arrow_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize RawBoost-augmented splits as Hugging Face Arrow."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--algorithm", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--input-column", default="input_values")
    parser.add_argument("--output-column", default="input_values")
    parser.add_argument("--num-proc", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--writer-batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fir-backend",
        choices=[item.value for item in FIRBackend],
        default=FIRBackend.OVERLAP_ADD.value,
    )
    parser.add_argument("--compute-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--load-from-cache-file",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = RawBoostConfig(
        seed=arguments.seed,
        fir_backend=arguments.fir_backend,
        compute_dtype=arguments.compute_dtype,
    )
    map_config = ArrowMapConfig(
        input_column=arguments.input_column,
        output_column=arguments.output_column,
        num_proc=arguments.num_proc,
        batch_size=arguments.batch_size,
        writer_batch_size=arguments.writer_batch_size,
        load_from_cache_file=arguments.load_from_cache_file,
    )
    destination = build_arrow_dataset(
        arguments.source,
        arguments.output,
        config,
        Algorithm(arguments.algorithm),
        map_config,
        splits=tuple(arguments.splits),
    )
    print(f"Saved augmented Arrow DatasetDict: {destination}")


if __name__ == "__main__":
    main()
