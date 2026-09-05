from __future__ import annotations

import json

import numpy as np
from datasets import Dataset, DatasetDict, List, Value, load_from_disk

from rawboost_opt import Algorithm, RawBoostConfig
from rawboost_opt.huggingface import ArrowMapConfig, augment_dataset, build_arrow_dataset


def _dataset(num_rows: int = 4, num_samples: int = 1_000) -> Dataset:
    waveforms = [
        np.linspace(-0.2, 0.2, num_samples, dtype=np.float32) + index * 1e-3
        for index in range(num_rows)
    ]
    return Dataset.from_dict(
        {
            "input_values": waveforms,
            "labels": [index % 2 for index in range(num_rows)],
        }
    ).cast_column("input_values", List(Value("float32")))


def test_augment_dataset_is_deterministic_across_process_counts() -> None:
    source = _dataset()
    config = RawBoostConfig(seed=18)
    serial = augment_dataset(
        source,
        config,
        Algorithm.ISD,
        ArrowMapConfig(batch_size=2, num_proc=None, load_from_cache_file=False),
        split_name="train",
    )
    parallel = augment_dataset(
        source,
        config,
        Algorithm.ISD,
        ArrowMapConfig(batch_size=2, num_proc=2, load_from_cache_file=False),
        split_name="train",
    )

    assert serial.features["input_values"] == List(Value("float32"))
    for index in range(len(source)):
        np.testing.assert_array_equal(
            np.asarray(serial[index]["input_values"]),
            np.asarray(parallel[index]["input_values"]),
        )


def test_build_arrow_dataset_preserves_unselected_split(tmp_path) -> None:
    source_path = tmp_path / "source"
    output_path = tmp_path / "augmented"
    source = DatasetDict({"train": _dataset(), "test": _dataset(2)})
    source.save_to_disk(source_path)

    build_arrow_dataset(
        source_path,
        output_path,
        RawBoostConfig(seed=5),
        Algorithm.ISD,
        ArrowMapConfig(batch_size=2, num_proc=None, load_from_cache_file=False),
        splits=("train",),
    )
    restored = load_from_disk(output_path)

    assert isinstance(restored, DatasetDict)
    assert restored["train"].features["input_values"] == List(Value("float32"))
    assert restored["test"][:] == source["test"][:]
    manifest = json.loads((output_path / "rawboost_manifest.json").read_text())
    assert manifest["algorithm"] == Algorithm.ISD.value
    assert manifest["augmented_splits"] == ["train"]
