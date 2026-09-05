from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import platform
import random
import shutil
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Every benchmark source is local. Avoid registry probes that add unrelated
# network latency to load_dataset("parquet").
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    DownloadConfig,
    IterableDataset,
    load_dataset,
    load_from_disk,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
# This benchmark lives under dev_functions/dataset_loading, so the repository
# root is two levels above the experiment directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_FIXED_DATASET = PROJECT_ROOT.parent / "datasets" / "ASVspoof2019_16k_4s_fixed"
DEFAULT_RAW_DATA = PROJECT_ROOT.parent / "datasets" / "ASVspoof2019" / "data"
SUPPORTED_VARIANTS = (
    "arrow",
    "parquet_cached_snappy",
    "parquet_cached_zstd",
    "parquet_streaming_snappy",
    "parquet_streaming_zstd",
    "raw_audio",
)


@dataclass(frozen=True)
class ArtifactManifest:
    fixed_dataset: str
    raw_data_dir: str
    split: str
    sample_count: int
    sample_rate: int
    target_samples: int
    seed: int
    parquet_shards: int
    source_fingerprint: str
    selected_indices: list[int]
    artifact_sizes_bytes: dict[str, int]


@dataclass(frozen=True)
class BenchmarkCase:
    variant: str
    batch_size: int
    num_workers: int
    repeat: int


@dataclass
class BenchmarkResult:
    variant: str
    batch_size: int
    num_workers: int
    repeat: int
    dataset_load_seconds: float
    iterator_setup_seconds: float
    first_batch_seconds: float
    measured_batches: int
    measured_samples: int
    steady_seconds: float
    samples_per_second: float
    logical_mib_per_second: float
    batch_wait_mean_ms: float
    batch_wait_p50_ms: float
    batch_wait_p95_ms: float
    checksum: float


class AudioClassificationCollator:
    """Local copy of the current training collator for isolated CPU benchmarks."""

    def __init__(
        self,
        input_values_column_name: str = "input_values",
        label_column_name: str = "labels",
    ) -> None:
        self.input_values_column_name = input_values_column_name
        self.label_column_name = label_column_name

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_values = []
        for feature in features:
            tensor = torch.as_tensor(
                feature[self.input_values_column_name],
                dtype=torch.float32,
            )
            input_values.append(tensor.reshape(-1))
        return {
            "input_values": torch.stack(input_values),
            "labels": torch.tensor(
                [int(feature[self.label_column_name]) for feature in features],
                dtype=torch.long,
            ),
        }


class RawAudioTransform:
    """Decode FLAC through the HF Audio/TorchCodec feature and produce fixed tensors."""

    def __init__(self, target_samples: int) -> None:
        self.target_samples = target_samples

    def _resize(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.reshape(-1).to(torch.float32)
        num_samples = waveform.numel()
        if num_samples == self.target_samples:
            return waveform
        if num_samples > self.target_samples:
            return waveform[: self.target_samples]
        if num_samples == 0:
            raise ValueError("Cannot benchmark an empty waveform.")
        repeats = math.ceil(self.target_samples / num_samples)
        return waveform.repeat(repeats)[: self.target_samples]

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        audio_values = batch["audio"]
        is_batched = isinstance(audio_values, list)
        decoders = audio_values if is_batched else [audio_values]
        waveforms = [
            self._resize(decoder.get_all_samples().data)
            for decoder in decoders
        ]
        batch = dict(batch)
        batch.pop("audio", None)
        batch["input_values"] = waveforms if is_batched else waveforms[0]
        return batch


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of non-negative integers.")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list.")
    unknown = set(values) - set(SUPPORTED_VARIANTS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown variants {sorted(unknown)}; supported: {list(SUPPORTED_VARIANTS)}"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare CPU-side loading throughput for ADD audio datasets.",
    )
    parser.add_argument(
        "command",
        choices=["prepare", "validate", "benchmark", "all"],
    )
    parser.add_argument("--fixed-dataset", type=Path, default=DEFAULT_FIXED_DATASET)
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--target-seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parquet-shards", type=int, default=8)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_ROOT / "results")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--variants",
        type=parse_str_list,
        default=list(SUPPORTED_VARIANTS),
    )
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[8, 32])
    parser.add_argument("--num-workers", type=parse_int_list, default=[0, 2, 4, 8])
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--measured-batches", type=int, default=64)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if args.sample_rate <= 0 or args.target_seconds <= 0:
        raise ValueError("sample_rate and target_seconds must be positive.")
    if args.parquet_shards <= 0:
        raise ValueError("parquet_shards must be positive.")
    if args.warmup_batches < 0 or args.measured_batches <= 0:
        raise ValueError("warmup_batches must be non-negative and measured_batches positive.")
    if args.validation_samples <= 0:
        raise ValueError("validation_samples must be positive.")
    if args.repeats <= 0 or args.prefetch_factor <= 0:
        raise ValueError("repeats and prefetch_factor must be positive.")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("batch_sizes must be positive.")


def resolved_artifact_dir(args: argparse.Namespace) -> Path:
    if args.artifact_dir is not None:
        return args.artifact_dir.expanduser().resolve()
    name = f"asv19_{args.split}_{args.sample_count}"
    return (EXPERIMENT_ROOT / "artifacts" / name).resolve()


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def load_split(path: Path, split: str) -> Dataset:
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split {split!r} not found in {path}; available: {list(dataset)}")
        dataset = dataset[split]
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset from {path}, got {type(dataset).__name__}.")
    return dataset


def select_source_examples(
    source: Dataset,
    sample_count: int,
    seed: int,
) -> tuple[Dataset, list[int]]:
    if sample_count > len(source):
        raise ValueError(
            f"sample_count={sample_count} exceeds split size {len(source)}."
        )
    indices = random.Random(seed).sample(range(len(source)), sample_count)
    selected = source.select(indices).select_columns(
        ["input_values", "labels", "utterance_id"],
    )
    return selected.with_format(None), indices


def write_parquet_shards(
    dataset: Dataset,
    output_dir: Path,
    *,
    compression: str,
    num_shards: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_shards = min(num_shards, len(dataset))
    for shard_index in range(actual_shards):
        output_path = output_dir / f"train-{shard_index:05d}-of-{actual_shards:05d}.parquet"
        shard = dataset.shard(
            num_shards=actual_shards,
            index=shard_index,
            contiguous=True,
        )
        table = shard.with_format("arrow")[:]
        if not isinstance(table, pa.Table):
            raise TypeError(
                f"Expected a PyArrow table while writing Parquet, got {type(table).__name__}."
            )
        pq.write_table(
            table,
            output_path,
            compression=compression,
            use_dictionary=False,
        )


def build_raw_manifest(
    selected: Dataset,
    raw_data_dir: Path,
    split: str,
) -> Dataset:
    utterance_ids = [str(value) for value in selected["utterance_id"]]
    labels = [int(value) for value in selected["labels"]]
    paths = [raw_data_dir / split / f"{utterance_id}.flac" for utterance_id in utterance_ids]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} selected raw audio files are missing; first entries: {preview}"
        )
    dataset = Dataset.from_dict(
        {
            # Keep paths as plain strings on disk. Casting to Audio before
            # save_to_disk embeds the FLAC bytes in Arrow and no longer measures
            # the original-file path.
            "audio_path": [str(path.resolve()) for path in paths],
            "labels": labels,
            "utterance_id": utterance_ids,
        }
    )
    return dataset


def prepare_artifacts(args: argparse.Namespace) -> Path:
    artifact_dir = resolved_artifact_dir(args)
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.is_file() and not args.rebuild:
        ensure_manifest_compatible(load_manifest(artifact_dir), args)
        logger.info("Reusing prepared artifacts: %s", artifact_dir)
        return artifact_dir
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)

    fixed_dataset = args.fixed_dataset.expanduser().resolve()
    raw_data_dir = args.raw_data_dir.expanduser().resolve()
    logger.info("Loading fixed source dataset: %s[%s]", fixed_dataset, args.split)
    source = load_split(fixed_dataset, args.split)
    selected, selected_indices = select_source_examples(
        source,
        args.sample_count,
        args.seed,
    )
    target_samples = round(args.sample_rate * args.target_seconds)
    first_length = len(selected[0]["input_values"])
    if first_length != target_samples:
        raise ValueError(
            f"Fixed source has {first_length} samples per item, expected {target_samples}."
        )

    arrow_dir = artifact_dir / "arrow"
    logger.info("Writing Arrow benchmark subset: %s", arrow_dir)
    selected.save_to_disk(
        str(arrow_dir),
        max_shard_size="256MB",
    )

    for compression in ("snappy", "zstd"):
        parquet_dir = artifact_dir / f"parquet_{compression}"
        logger.info("Writing %s Parquet shards: %s", compression, parquet_dir)
        write_parquet_shards(
            selected,
            parquet_dir,
            compression=compression,
            num_shards=args.parquet_shards,
        )

    raw_manifest_dir = artifact_dir / "raw_manifest"
    logger.info("Writing raw FLAC path manifest: %s", raw_manifest_dir)
    raw_manifest = build_raw_manifest(
        selected,
        raw_data_dir,
        args.split,
    )
    raw_manifest.save_to_disk(str(raw_manifest_dir))

    sizes = {
        "arrow": directory_size(arrow_dir),
        "parquet_snappy": directory_size(artifact_dir / "parquet_snappy"),
        "parquet_zstd": directory_size(artifact_dir / "parquet_zstd"),
        "raw_manifest": directory_size(raw_manifest_dir),
        "raw_flac_selected": sum(
            (raw_data_dir / args.split / f"{utterance_id}.flac").stat().st_size
            for utterance_id in selected["utterance_id"]
        ),
    }
    manifest = ArtifactManifest(
        fixed_dataset=str(fixed_dataset),
        raw_data_dir=str(raw_data_dir),
        split=args.split,
        sample_count=args.sample_count,
        sample_rate=args.sample_rate,
        target_samples=target_samples,
        seed=args.seed,
        parquet_shards=args.parquet_shards,
        source_fingerprint=source._fingerprint,
        selected_indices=selected_indices,
        artifact_sizes_bytes=sizes,
    )
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Prepared artifacts: %s", artifact_dir)
    return artifact_dir


def load_manifest(artifact_dir: Path) -> ArtifactManifest:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run the prepare or all command first."
        )
    return ArtifactManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))


def ensure_manifest_compatible(
    manifest: ArtifactManifest,
    args: argparse.Namespace,
) -> None:
    expected = {
        "fixed_dataset": str(args.fixed_dataset.expanduser().resolve()),
        "raw_data_dir": str(args.raw_data_dir.expanduser().resolve()),
        "split": args.split,
        "sample_count": args.sample_count,
        "sample_rate": args.sample_rate,
        "target_samples": round(args.sample_rate * args.target_seconds),
        "seed": args.seed,
        "parquet_shards": args.parquet_shards,
    }
    mismatches = {
        name: {"artifact": getattr(manifest, name), "requested": value}
        for name, value in expected.items()
        if getattr(manifest, name) != value
    }
    if mismatches:
        details = ", ".join(
            f"{name}={values['artifact']!r}->{values['requested']!r}"
            for name, values in mismatches.items()
        )
        raise ValueError(
            f"Prepared artifact settings do not match this command: {details}. "
            "Use --rebuild or select a different --artifact-dir."
        )


def parquet_files(artifact_dir: Path, compression: str) -> list[str]:
    files = sorted((artifact_dir / f"parquet_{compression}").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No {compression} Parquet shards under {artifact_dir}."
        )
    return [str(path) for path in files]


def format_fixed_dataset(dataset: Dataset) -> Dataset:
    return dataset.with_format(
        "torch",
        columns=["input_values", "labels"],
        output_all_columns=False,
    )


def load_variant_dataset(
    variant: str,
    artifact_dir: Path,
    manifest: ArtifactManifest,
    *,
    shuffle: bool,
    seed: int,
) -> Dataset | IterableDataset:
    if variant == "arrow":
        return format_fixed_dataset(load_split(artifact_dir / "arrow", "train"))

    if variant.startswith("parquet_cached_"):
        compression = variant.removeprefix("parquet_cached_")
        dataset = load_dataset(
            "parquet",
            data_files={"train": parquet_files(artifact_dir, compression)},
            split="train",
            cache_dir=str(artifact_dir / "hf_parquet_cache" / compression),
            download_config=DownloadConfig(local_files_only=True),
        )
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected map-style Parquet Dataset, got {type(dataset).__name__}.")
        return format_fixed_dataset(dataset)

    if variant.startswith("parquet_streaming_"):
        compression = variant.removeprefix("parquet_streaming_")
        files = parquet_files(artifact_dir, compression)
        if shuffle:
            # The artifact rows are already selected in random order. Shuffling
            # file order preserves one iterable shard per Parquet file, whereas
            # IterableDataset.shuffle() collapses HF Datasets 5.0 to one shard
            # and prevents DataLoader workers from reading in parallel.
            random.Random(seed).shuffle(files)
        dataset = load_dataset(
            "parquet",
            data_files={"train": files},
            split="train",
            streaming=True,
            download_config=DownloadConfig(local_files_only=True),
        )
        if not isinstance(dataset, IterableDataset):
            raise TypeError(f"Expected IterableDataset, got {type(dataset).__name__}.")
        return dataset.with_format("torch")

    if variant == "raw_audio":
        dataset = load_split(artifact_dir / "raw_manifest", "train")
        dataset = dataset.rename_column("audio_path", "audio")
        dataset = dataset.cast_column(
            "audio",
            Audio(sampling_rate=manifest.sample_rate, decode=True),
        )
        return dataset.with_transform(RawAudioTransform(manifest.target_samples))

    raise ValueError(f"Unsupported variant: {variant}")


def dataset_head(
    dataset: Dataset | IterableDataset,
    count: int,
) -> list[dict[str, Any]]:
    if isinstance(dataset, Dataset):
        return [dataset[index] for index in range(count)]
    return list(dataset.take(count))


def validate_variant_outputs(
    args: argparse.Namespace,
    artifact_dir: Path,
) -> Path:
    """Verify that every storage path returns the same labels and waveforms."""

    manifest = load_manifest(artifact_dir)
    ensure_manifest_compatible(manifest, args)
    count = min(args.validation_samples, manifest.sample_count)
    reference_dataset = load_variant_dataset(
        "arrow",
        artifact_dir,
        manifest,
        shuffle=False,
        seed=args.seed,
    )
    reference_rows = dataset_head(reference_dataset, count)
    results = []

    for variant in args.variants:
        dataset = load_variant_dataset(
            variant,
            artifact_dir,
            manifest,
            shuffle=False,
            seed=args.seed,
        )
        rows = dataset_head(dataset, count)
        if len(rows) != count:
            raise AssertionError(
                f"{variant} returned {len(rows)} validation rows, expected {count}."
            )

        exact_waveforms = True
        matching_labels = True
        matching_shapes = True
        max_abs_error = 0.0
        max_rmse = 0.0
        for reference, candidate in zip(reference_rows, rows, strict=True):
            reference_values = torch.as_tensor(
                reference["input_values"],
                dtype=torch.float32,
            ).reshape(-1)
            candidate_values = torch.as_tensor(
                candidate["input_values"],
                dtype=torch.float32,
            ).reshape(-1)
            matching_labels &= int(reference["labels"]) == int(candidate["labels"])
            if reference_values.shape != candidate_values.shape:
                matching_shapes = False
                exact_waveforms = False
                continue
            difference = reference_values - candidate_values
            exact_waveforms &= torch.equal(reference_values, candidate_values)
            max_abs_error = max(
                max_abs_error,
                float(difference.abs().max().item()),
            )
            max_rmse = max(
                max_rmse,
                float(torch.sqrt(torch.mean(difference.square())).item()),
            )

        passed = matching_labels and matching_shapes and exact_waveforms
        result = {
            "variant": variant,
            "samples": count,
            "matching_labels": matching_labels,
            "matching_shapes": matching_shapes,
            "exact_waveforms": exact_waveforms,
            "max_abs_error": max_abs_error,
            "max_rmse": max_rmse,
            "passed": passed,
        }
        results.append(result)
        logger.info(
            "Validated %-28s exact=%s max_abs=%.3g max_rmse=%.3g",
            variant,
            passed,
            max_abs_error,
            max_rmse,
        )
        del dataset
        gc.collect()

    output_dir = args.results_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"dataset-validation-{timestamp}.json"
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "artifact_dir": str(artifact_dir),
        "reference": "arrow",
        "samples": count,
        "target_samples": manifest.target_samples,
        "results": results,
        "all_passed": all(result["passed"] for result in results),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not payload["all_passed"]:
        raise AssertionError(f"Dataset output validation failed; see {output_path}.")
    logger.info("Wrote validation results: %s", output_path)
    return output_path


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def make_dataloader(
    dataset: Dataset | IterableDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    seed: int,
) -> DataLoader:
    is_iterable = isinstance(dataset, IterableDataset)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle and not is_iterable,
        "num_workers": num_workers,
        "collate_fn": AudioClassificationCollator(),
        "pin_memory": False,
        "drop_last": True,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def consume_batch(batch: dict[str, torch.Tensor]) -> tuple[int, int, float]:
    input_values = batch["input_values"]
    labels = batch["labels"]
    if input_values.device.type != "cpu":
        raise RuntimeError("Dataset benchmark unexpectedly produced a non-CPU tensor.")
    logical_bytes = input_values.numel() * input_values.element_size()
    logical_bytes += labels.numel() * labels.element_size()
    checksum = float(input_values[:, 0].sum().item() + labels.sum().item())
    return input_values.shape[0], logical_bytes, checksum


def benchmark_case(
    dataset: Dataset | IterableDataset,
    case: BenchmarkCase,
    *,
    dataset_load_seconds: float,
    sample_count: int,
    warmup_batches: int,
    requested_measured_batches: int,
    shuffle: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    seed: int,
) -> BenchmarkResult:
    available_batches = sample_count // case.batch_size
    measured_batches = min(
        requested_measured_batches,
        available_batches - warmup_batches - 1,
    )
    if measured_batches <= 0:
        raise ValueError(
            f"Not enough samples for batch_size={case.batch_size}, "
            f"warmup_batches={warmup_batches}, and one measured batch."
        )

    loader = make_dataloader(
        dataset,
        batch_size=case.batch_size,
        num_workers=case.num_workers,
        shuffle=shuffle,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=seed + case.repeat,
    )
    iterator_start = time.perf_counter()
    iterator = iter(loader)
    iterator_setup_seconds = time.perf_counter() - iterator_start

    first_start = time.perf_counter()
    first_batch = next(iterator)
    first_batch_seconds = time.perf_counter() - first_start
    _, _, checksum = consume_batch(first_batch)

    for _ in range(warmup_batches):
        batch = next(iterator)
        _, _, batch_checksum = consume_batch(batch)
        checksum += batch_checksum

    wait_times = []
    measured_samples = 0
    logical_bytes = 0
    steady_start = time.perf_counter()
    for _ in range(measured_batches):
        batch_start = time.perf_counter()
        batch = next(iterator)
        wait_times.append(time.perf_counter() - batch_start)
        batch_samples, batch_bytes, batch_checksum = consume_batch(batch)
        measured_samples += batch_samples
        logical_bytes += batch_bytes
        checksum += batch_checksum
    steady_seconds = time.perf_counter() - steady_start

    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown_workers):
        shutdown_workers()
    del iterator, loader
    gc.collect()

    return BenchmarkResult(
        variant=case.variant,
        batch_size=case.batch_size,
        num_workers=case.num_workers,
        repeat=case.repeat,
        dataset_load_seconds=dataset_load_seconds,
        iterator_setup_seconds=iterator_setup_seconds,
        first_batch_seconds=first_batch_seconds,
        measured_batches=measured_batches,
        measured_samples=measured_samples,
        steady_seconds=steady_seconds,
        samples_per_second=measured_samples / steady_seconds,
        logical_mib_per_second=(logical_bytes / (1024**2)) / steady_seconds,
        batch_wait_mean_ms=statistics.fmean(wait_times) * 1_000.0,
        batch_wait_p50_ms=percentile(wait_times, 0.50) * 1_000.0,
        batch_wait_p95_ms=percentile(wait_times, 0.95) * 1_000.0,
        checksum=checksum,
    )


def aggregate_results(results: Sequence[BenchmarkResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[BenchmarkResult]] = {}
    for result in results:
        key = (result.variant, result.batch_size, result.num_workers)
        grouped.setdefault(key, []).append(result)

    rows = []
    for (variant, batch_size, num_workers), values in grouped.items():
        rows.append(
            {
                "variant": variant,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "repeats": len(values),
                "dataset_load_seconds": statistics.median(
                    value.dataset_load_seconds for value in values
                ),
                "first_batch_seconds": statistics.median(
                    value.first_batch_seconds for value in values
                ),
                "samples_per_second": statistics.median(
                    value.samples_per_second for value in values
                ),
                "logical_mib_per_second": statistics.median(
                    value.logical_mib_per_second for value in values
                ),
                "batch_wait_p50_ms": statistics.median(
                    value.batch_wait_p50_ms for value in values
                ),
                "batch_wait_p95_ms": statistics.median(
                    value.batch_wait_p95_ms for value in values
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row["batch_size"]),
            int(row["num_workers"]),
            -float(row["samples_per_second"]),
        ),
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_bytes(value: int) -> str:
    return f"{value / (1024**2):.1f} MiB"


def write_markdown_report(
    path: Path,
    manifest: ArtifactManifest,
    summary: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Dataset loading benchmark",
        "",
        "## Environment",
        "",
        f"- CPU: `{metadata['cpu']}`",
        f"- Python: `{metadata['python_version']}`",
        f"- PyTorch: `{metadata['torch_version']}`",
        f"- datasets: `{metadata['datasets_version']}`",
        f"- pyarrow: `{metadata['pyarrow_version']}`",
        f"- Samples: `{manifest.sample_count}` x `{manifest.target_samples}` float32 values",
        f"- Shuffle: `{metadata['shuffle']}`",
        "",
        "## Artifact sizes",
        "",
        "| Artifact | Size |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {format_bytes(size)} |"
        for name, size in manifest.artifact_sizes_bytes.items()
    )
    lines.extend(
        [
            "",
            "## Steady-state results",
            "",
            "| Variant | Batch | Workers | Dataset load (s) | First batch (s) | Samples/s | Logical MiB/s | p50 wait (ms) | p95 wait (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            "| {variant} | {batch_size} | {num_workers} | "
            "{dataset_load_seconds:.3f} | {first_batch_seconds:.3f} | "
            "{samples_per_second:.1f} | {logical_mib_per_second:.1f} | "
            "{batch_wait_p50_ms:.2f} | {batch_wait_p95_ms:.2f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def runtime_metadata(args: argparse.Namespace) -> dict[str, Any]:
    import datasets
    import pyarrow

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "cpu": cpu_model_name(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "datasets_version": datasets.__version__,
        "pyarrow_version": pyarrow.__version__,
        "shuffle": args.shuffle,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "warmup_batches": args.warmup_batches,
        "requested_measured_batches": args.measured_batches,
    }


def cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def run_benchmarks(args: argparse.Namespace, artifact_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    manifest = load_manifest(artifact_dir)
    ensure_manifest_compatible(manifest, args)
    all_results = []
    for variant in args.variants:
        load_start = time.perf_counter()
        dataset = load_variant_dataset(
            variant,
            artifact_dir,
            manifest,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        dataset_load_seconds = time.perf_counter() - load_start
        logger.info(
            "Loaded variant=%s in %.3fs",
            variant,
            dataset_load_seconds,
        )
        for batch_size in args.batch_sizes:
            for num_workers in args.num_workers:
                for repeat in range(args.repeats):
                    case = BenchmarkCase(
                        variant=variant,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        repeat=repeat,
                    )
                    logger.info(
                        "Benchmark %s batch=%s workers=%s repeat=%s",
                        variant,
                        batch_size,
                        num_workers,
                        repeat,
                    )
                    result = benchmark_case(
                        dataset,
                        case,
                        dataset_load_seconds=dataset_load_seconds,
                        sample_count=manifest.sample_count,
                        warmup_batches=args.warmup_batches,
                        requested_measured_batches=args.measured_batches,
                        shuffle=args.shuffle,
                        persistent_workers=args.persistent_workers,
                        prefetch_factor=args.prefetch_factor,
                        seed=args.seed,
                    )
                    all_results.append(result)
                    logger.info(
                        "  %.1f samples/s, first batch %.3fs, p95 wait %.2fms",
                        result.samples_per_second,
                        result.first_batch_seconds,
                        result.batch_wait_p95_ms,
                    )
        del dataset
        gc.collect()

    metadata = runtime_metadata(args)
    summary = aggregate_results(all_results)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = args.results_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"dataset-loading-{timestamp}.json"
    csv_path = output_dir / f"dataset-loading-{timestamp}.csv"
    report_path = output_dir / f"dataset-loading-{timestamp}.md"
    payload = {
        "metadata": metadata,
        "manifest": asdict(manifest),
        "arguments": vars(args) | {"artifact_dir": str(artifact_dir)},
        "raw_results": [asdict(result) for result in all_results],
        "summary": summary,
    }
    payload["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload["arguments"].items()
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(csv_path, summary)
    write_markdown_report(report_path, manifest, summary, metadata)
    logger.info("Wrote results: %s", json_path)
    logger.info("Wrote summary: %s", report_path)
    return report_path, summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    artifact_dir = resolved_artifact_dir(args)

    if args.command in {"prepare", "all"}:
        artifact_dir = prepare_artifacts(args)
    if args.command in {"validate", "all"}:
        validate_variant_outputs(args, artifact_dir)
    if args.command in {"benchmark", "all"}:
        run_benchmarks(args, artifact_dir)


if __name__ == "__main__":
    main()
