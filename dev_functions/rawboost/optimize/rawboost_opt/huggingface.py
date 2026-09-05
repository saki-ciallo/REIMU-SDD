from __future__ import annotations

import json
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, List, Value, load_from_disk

from .api import RawBoost
from .config import Algorithm, RawBoostConfig

_WORKER_AUGMENTERS: dict[RawBoostConfig, RawBoost] = {}


def _worker_augmenter(config: RawBoostConfig) -> RawBoost:
    augmenter = _WORKER_AUGMENTERS.get(config)
    if augmenter is None:
        augmenter = RawBoost(config)
        _WORKER_AUGMENTERS[config] = augmenter
    return augmenter


def _sample_seed(base_seed: int, split_salt: int, index: int) -> int:
    sequence = np.random.SeedSequence([base_seed, split_salt, index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


@dataclass(frozen=True, slots=True)
class ArrowMapConfig:
    input_column: str = "input_values"
    output_column: str = "input_values"
    batch_size: int = 32
    writer_batch_size: int = 100
    num_proc: int | None = None
    load_from_cache_file: bool = True

    def __post_init__(self) -> None:
        if not self.input_column or not self.output_column:
            raise ValueError("input_column and output_column must not be empty.")
        if self.batch_size <= 0 or self.writer_batch_size <= 0:
            raise ValueError("batch_size and writer_batch_size must be positive.")
        if self.num_proc is not None and self.num_proc <= 0:
            raise ValueError("num_proc must be positive or null.")


@dataclass(frozen=True, slots=True)
class HFRawBoostMapper:
    rawboost_config: RawBoostConfig
    algorithm: Algorithm
    input_column: str
    output_column: str
    split_salt: int

    def __call__(
        self,
        batch: dict[str, list[object]],
        indices: list[int],
    ) -> dict[str, list[np.ndarray]]:
        augmenter = _worker_augmenter(self.rawboost_config)
        outputs = [
            augmenter.augment(
                waveform,
                self.algorithm,
                seed=_sample_seed(self.rawboost_config.seed, self.split_salt, index),
                backend="numpy",
                output_type="numpy",
            )
            for waveform, index in zip(batch[self.input_column], indices, strict=True)
        ]
        return {self.output_column: outputs}


def augment_dataset(
    dataset: Dataset,
    rawboost_config: RawBoostConfig,
    algorithm: Algorithm | int,
    map_config: ArrowMapConfig,
    *,
    split_name: str,
) -> Dataset:
    """Materialize deterministic float32 augmentation into an Arrow Dataset."""

    if map_config.input_column not in dataset.column_names:
        raise KeyError(f"Missing input column {map_config.input_column!r}.")
    algorithm = Algorithm(algorithm)
    # Arrow delivers a float32 ndarray directly, avoiding list -> ndarray copies.
    plain_dataset = dataset.with_format(
        "numpy",
        columns=[map_config.input_column],
        output_all_columns=True,
    )
    mapper = HFRawBoostMapper(
        rawboost_config=rawboost_config,
        algorithm=algorithm,
        input_column=map_config.input_column,
        output_column=map_config.output_column,
        split_salt=zlib.crc32(split_name.encode()),
    )
    mapped = plain_dataset.map(
        mapper,
        batched=True,
        with_indices=True,
        batch_size=map_config.batch_size,
        writer_batch_size=map_config.writer_batch_size,
        num_proc=map_config.num_proc,
        load_from_cache_file=map_config.load_from_cache_file,
        desc=f"RawBoost algo {algorithm.value} [{split_name}]",
    )
    return mapped.cast_column(map_config.output_column, List(Value("float32")))


def build_arrow_dataset(
    source_path: str | Path,
    output_path: str | Path,
    rawboost_config: RawBoostConfig,
    algorithm: Algorithm | int,
    map_config: ArrowMapConfig,
    *,
    splits: tuple[str, ...] = ("train", "validation"),
) -> Path:
    """Augment selected splits and save a Hugging Face DatasetDict as Arrow."""

    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}. Choose a new directory.")
    loaded = load_from_disk(str(source_path))
    if not isinstance(loaded, DatasetDict):
        raise TypeError("source_path must contain a Hugging Face DatasetDict.")
    missing = set(splits) - set(loaded)
    if missing:
        raise KeyError(f"Missing requested splits: {sorted(missing)}")

    started = time.perf_counter()
    output_splits = {
        name: (
            augment_dataset(
                dataset,
                rawboost_config,
                algorithm,
                map_config,
                split_name=name,
            )
            if name in splits
            else dataset.with_format(None)
        )
        for name, dataset in loaded.items()
    }
    result = DatasetDict(output_splits)
    result.save_to_disk(str(output_path), num_proc=map_config.num_proc)
    manifest = {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "algorithm": Algorithm(algorithm).value,
        "augmented_splits": list(splits),
        "split_sizes": {name: len(dataset) for name, dataset in result.items()},
        "rawboost_config": {
            **asdict(rawboost_config),
            "fir_backend": rawboost_config.fir_backend.value,
        },
        "map_config": asdict(map_config),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_path / "rawboost_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path
