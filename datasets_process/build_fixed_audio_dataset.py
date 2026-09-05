from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from datasets import ClassLabel, DatasetDict, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets_load import AudioTransformConfig, load_audio_dataset  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetPreset:
    data_dir: Path
    output_dir: Path
    exception_audio_files: tuple[str, ...] = ()


DATASET_PRESETS = {
    "asv19": DatasetPreset(
        data_dir=DATASETS_ROOT / "ASVspoof2019_hf" / "data",
        output_dir=DATASETS_ROOT / "ASVspoof2019_16k_4s_fixed",
    ),
    "asv21-la": DatasetPreset(
        data_dir=DATASETS_ROOT / "ASVspoof2021_LA_hf" / "data",
        output_dir=DATASETS_ROOT / "ASVspoof2021_LA_16k_4s_fixed",
    ),
    "asv21-df": DatasetPreset(
        data_dir=DATASETS_ROOT / "ASVspoof2021_DF_hf" / "data",
        output_dir=DATASETS_ROOT / "ASVspoof2021_DF_16k_4s_fixed",
        exception_audio_files=(
            "DF_E_2416160",
            "DF_E_2101080",
            "DF_E_4830191",
            "DF_E_4459808",
            "DF_E_4887195",
        ),
    ),
}


@dataclass(frozen=True)
class FixedDatasetBuildConfig:
    data_dir: Path
    output_dir: Path
    sampling_rate: int
    target_seconds: float
    label_column: str
    label_names: tuple[str, ...]
    exception_audio_files: tuple[str, ...]
    verify_examples_per_split: int

    @property
    def target_length(self) -> int:
        return int(round(self.sampling_rate * self.target_seconds))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize an audiofolder dataset as fixed-length ADD input_values "
            "and save it with datasets.save_to_disk."
        )
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DATASET_PRESETS),
        help="Use the known source/output paths and exception list for an ASV dataset.",
    )
    parser.add_argument(
        "--data-dir",
        help="audiofolder data directory. Overrides the selected preset.",
    )
    parser.add_argument(
        "--output-dir",
        help="save_to_disk destination. Overrides the selected preset.",
    )
    parser.add_argument("--sampling-rate", type=int, default=16_000)
    parser.add_argument("--target-seconds", type=float, default=4.0)
    parser.add_argument("--label-column", default="labels")
    parser.add_argument(
        "--label-names",
        nargs="+",
        default=["bonafide", "spoof"],
        help="ClassLabel names in integer-label order.",
    )
    parser.add_argument(
        "--exception-audio-file",
        action="append",
        default=[],
        help=(
            "Path, filename, or utterance_id allowed to fall back to ffmpeg after "
            "TorchCodec fails. May be repeated."
        ),
    )
    parser.add_argument(
        "--verify-examples-per-split",
        type=int,
        default=3,
        help="Number of evenly spaced examples checked after save/reload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved build configuration without loading audio.",
    )
    return parser.parse_args(argv)


def resolve_build_config(args: argparse.Namespace) -> FixedDatasetBuildConfig:
    preset = DATASET_PRESETS.get(args.preset) if args.preset is not None else None
    data_dir_value = args.data_dir if args.data_dir is not None else (
        preset.data_dir if preset is not None else None
    )
    output_dir_value = args.output_dir if args.output_dir is not None else (
        preset.output_dir if preset is not None else None
    )
    if data_dir_value is None or output_dir_value is None:
        raise ValueError(
            "Provide --preset, or provide both --data-dir and --output-dir."
        )
    if (
        preset is not None
        and args.output_dir is None
        and (args.sampling_rate != 16_000 or args.target_seconds != 4.0)
    ):
        raise ValueError(
            "Preset output directories are named for 16 kHz / 4 s data. "
            "Provide --output-dir when changing sampling_rate or target_seconds."
        )

    data_dir = Path(data_dir_value).expanduser().resolve()
    output_dir = Path(output_dir_value).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Audio data directory does not exist: {data_dir}")
    if output_dir.exists() and not args.dry_run:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new path or remove the old dataset explicitly."
        )
    if args.sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    if args.target_seconds <= 0:
        raise ValueError("target_seconds must be positive.")
    if args.verify_examples_per_split < 0:
        raise ValueError("verify_examples_per_split must be non-negative.")
    if not args.label_column:
        raise ValueError("label_column must not be empty.")
    if len(args.label_names) < 2 or len(set(args.label_names)) != len(args.label_names):
        raise ValueError("label_names must contain at least two unique names.")

    preset_exceptions = preset.exception_audio_files if preset is not None else ()
    exceptions = tuple(dict.fromkeys((*preset_exceptions, *args.exception_audio_file)))
    return FixedDatasetBuildConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        sampling_rate=args.sampling_rate,
        target_seconds=args.target_seconds,
        label_column=args.label_column,
        label_names=tuple(args.label_names),
        exception_audio_files=exceptions,
        verify_examples_per_split=args.verify_examples_per_split,
    )


def _verification_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return sorted(
        {
            round(index * (length - 1) / (count - 1))
            for index in range(count)
        }
    )


def _label_summary(dataset: DatasetDict, label_column: str) -> dict[str, dict[int, int]]:
    summaries: dict[str, dict[int, int]] = {}
    for split_name, split in dataset.items():
        counts = Counter(int(value) for value in split[label_column])
        summaries[split_name] = dict(sorted(counts.items()))
    return summaries


def verify_fixed_dataset(
    dataset: DatasetDict,
    config: FixedDatasetBuildConfig,
) -> None:
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(dataset).__name__}.")
    if not dataset:
        raise ValueError("DatasetDict does not contain any splits.")

    for split_name, split in dataset.items():
        required_columns = {"input_values", config.label_column}
        missing_columns = required_columns.difference(split.column_names)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Split {split_name!r} is missing columns: {missing}.")

        label_feature = split.features[config.label_column]
        if not isinstance(label_feature, ClassLabel):
            raise TypeError(
                f"Split {split_name!r} label feature must be ClassLabel, "
                f"got {type(label_feature).__name__}."
            )
        if tuple(label_feature.names) != config.label_names:
            raise ValueError(
                f"Split {split_name!r} label names {label_feature.names} do not match "
                f"{list(config.label_names)}."
            )

        for index in _verification_indices(
            len(split),
            config.verify_examples_per_split,
        ):
            waveform = torch.as_tensor(split[index]["input_values"])
            if waveform.ndim != 1:
                raise ValueError(
                    f"Split {split_name!r} example {index} input_values must be 1D, "
                    f"got shape {tuple(waveform.shape)}."
                )
            if waveform.numel() != config.target_length:
                raise ValueError(
                    f"Split {split_name!r} example {index} has {waveform.numel()} samples; "
                    f"expected {config.target_length}."
                )
            if not waveform.is_floating_point():
                raise TypeError(
                    f"Split {split_name!r} example {index} input_values must be floating point."
                )
            if not torch.isfinite(waveform).all():
                raise FloatingPointError(
                    f"Split {split_name!r} example {index} contains non-finite samples."
                )


def build_fixed_audio_dataset(config: FixedDatasetBuildConfig) -> DatasetDict:
    """Materialize, save, reload, and verify one fixed-length audio dataset."""

    transform_config = AudioTransformConfig(
        sampling_rate=config.sampling_rate,
        target_seconds=config.target_seconds,
        trim_select="fixed",
        label_column=config.label_column,
        label_names=list(config.label_names),
        use_dataset_cache=True,
        exception_audio_files=list(config.exception_audio_files),
    )
    logger.info("Loading audiofolder dataset from %s", config.data_dir)
    dataset = load_audio_dataset(
        data_dir=str(config.data_dir),
        config=transform_config,
    )
    verify_fixed_dataset(dataset, config)

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving fixed dataset to %s", config.output_dir)
    dataset.save_to_disk(str(config.output_dir))

    logger.info("Reloading saved dataset for verification")
    reloaded = load_from_disk(str(config.output_dir))
    if not isinstance(reloaded, DatasetDict):
        raise TypeError(
            f"Expected reloaded DatasetDict, got {type(reloaded).__name__}."
        )
    expected_lengths = {name: len(split) for name, split in dataset.items()}
    reloaded_lengths = {name: len(split) for name, split in reloaded.items()}
    if reloaded_lengths != expected_lengths:
        raise RuntimeError(
            f"Split lengths changed after save/reload: "
            f"{expected_lengths} != {reloaded_lengths}."
        )
    verify_fixed_dataset(reloaded, config)
    return reloaded


def _format_build_config(config: FixedDatasetBuildConfig) -> str:
    return "\n".join(
        [
            f"data_dir: {config.data_dir}",
            f"output_dir: {config.output_dir}",
            f"sampling_rate: {config.sampling_rate}",
            f"target_seconds: {config.target_seconds}",
            f"target_length: {config.target_length}",
            f"label_column: {config.label_column}",
            f"label_names: {list(config.label_names)}",
            f"exception_audio_files: {list(config.exception_audio_files)}",
            f"verify_examples_per_split: {config.verify_examples_per_split}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    config = resolve_build_config(args)
    logger.info("Resolved fixed-dataset build:\n%s", _format_build_config(config))
    if args.dry_run:
        return

    dataset = build_fixed_audio_dataset(config)
    logger.info("Saved splits: %s", {name: len(split) for name, split in dataset.items()})
    logger.info("Label counts: %s", _label_summary(dataset, config.label_column))


if __name__ == "__main__":
    main()


__all__ = [
    "DATASET_PRESETS",
    "DatasetPreset",
    "FixedDatasetBuildConfig",
    "build_fixed_audio_dataset",
    "resolve_build_config",
    "verify_fixed_dataset",
]
