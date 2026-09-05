from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_DIR = PROJECT_ROOT.parent / "datasets"
DEFAULT_KEYS_DIR = (
    PROJECT_ROOT / "official_scores" / "2021" / "eval-package" / "keys"
)

METADATA_FIELDS = (
    "file_name",
    "utterance_id",
    "system_id",
    "labels",
)


@dataclass(frozen=True)
class TrialRecord:
    utterance_id: str
    system_id: str
    label: str

    @property
    def audio_filename(self) -> str:
        return f"{self.utterance_id}.flac"


@dataclass(frozen=True)
class SubsetSpec:
    name: str
    source_dirname: str
    output_dirname: str


@dataclass(frozen=True)
class PreparedSubset:
    spec: SubsetSpec
    protocol_path: Path
    source_audio_dir: Path
    output_dir: Path
    records: tuple[TrialRecord, ...]
    audio_count: int
    extra_audio_count: int


SUBSET_SPECS = {
    "DF": SubsetSpec(
        name="DF",
        source_dirname="ASVspoof2021_DF_eval",
        output_dirname="ASVspoof2021_DF_hf",
    ),
    "LA": SubsetSpec(
        name="LA",
        source_dirname="ASVspoof2021_LA_eval",
        output_dirname="ASVspoof2021_LA_hf",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Hugging Face audiofolder datasets for the ASVspoof2021 DF and "
            "LA evaluation sets."
        )
    )
    parser.add_argument(
        "--subset",
        choices=("DF", "LA", "both"),
        default="both",
        help="Dataset subset to prepare. Default: both.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help=(
            "Directory containing ASVspoof2021_DF_eval and ASVspoof2021_LA_eval. "
            f"Default: {DEFAULT_DATASETS_DIR}"
        ),
    )
    parser.add_argument(
        "--keys-dir",
        type=Path,
        default=DEFAULT_KEYS_DIR,
        help=(
            "Official ASVspoof2021 eval-package keys directory. "
            f"Default: {DEFAULT_KEYS_DIR}"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "move", "copy"),
        default="hardlink",
        help=(
            "Audio transfer method. hardlink preserves the source without duplicating "
            "audio data; move renames the flac directory; copy duplicates the files."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and metadata without creating output datasets.",
    )
    return parser.parse_args(argv)


def read_trials(protocol_path: Path) -> tuple[TrialRecord, ...]:
    """Read every row from an ASVspoof2021 trial-metadata file."""

    records: list[TrialRecord] = []
    seen_utterance_ids: set[str] = set()
    with protocol_path.open(encoding="utf-8") as protocol_file:
        for line_number, raw_line in enumerate(protocol_file, start=1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) < 8:
                raise ValueError(
                    f"{protocol_path}:{line_number} must contain at least 8 fields, "
                    f"got {len(fields)}."
                )
            utterance_id = fields[1]
            system_id = fields[4]
            label = fields[5]
            if label not in {"bonafide", "spoof"}:
                raise ValueError(
                    f"{protocol_path}:{line_number} has unsupported label {label!r}."
                )
            if utterance_id in seen_utterance_ids:
                raise ValueError(
                    f"{protocol_path}:{line_number} repeats utterance_id "
                    f"{utterance_id!r}."
                )
            seen_utterance_ids.add(utterance_id)
            records.append(
                TrialRecord(
                    utterance_id=utterance_id,
                    system_id=system_id,
                    label=label,
                )
            )

    if not records:
        raise ValueError(f"No rows found in trial metadata: {protocol_path}")
    return tuple(records)


def inspect_subset(
    spec: SubsetSpec,
    datasets_dir: Path,
    keys_dir: Path,
) -> PreparedSubset:
    """Resolve one subset and verify every trial has a source FLAC."""

    protocol_path = keys_dir / spec.name / "CM" / "trial_metadata.txt"
    source_audio_dir = datasets_dir / spec.source_dirname / "flac"
    output_dir = datasets_dir / spec.output_dirname
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing trial metadata: {protocol_path}")
    if not source_audio_dir.is_dir():
        raise FileNotFoundError(f"Missing source audio directory: {source_audio_dir}")

    records = read_trials(protocol_path)
    audio_filenames = {
        path.name
        for path in source_audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".flac"
    }
    expected_filenames = {record.audio_filename for record in records}
    missing_filenames = expected_filenames.difference(audio_filenames)
    if missing_filenames:
        examples = ", ".join(sorted(missing_filenames)[:5])
        raise FileNotFoundError(
            f"{spec.name} is missing {len(missing_filenames)} trial audio files. "
            f"Examples: {examples}"
        )

    return PreparedSubset(
        spec=spec,
        protocol_path=protocol_path,
        source_audio_dir=source_audio_dir,
        output_dir=output_dir,
        records=records,
        audio_count=len(audio_filenames),
        extra_audio_count=len(audio_filenames - expected_filenames),
    )


def write_metadata_csv(
    output_split_dir: Path,
    records: Sequence[TrialRecord],
) -> Path:
    """Write ASVspoof2021 metadata in Hugging Face audiofolder format."""

    metadata_path = output_split_dir / "metadata.csv"
    temporary_path = output_split_dir / "metadata.csv.tmp"
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "file_name": record.audio_filename,
                    "utterance_id": record.utterance_id,
                    "system_id": record.system_id,
                    "labels": record.label,
                }
            )
    temporary_path.replace(metadata_path)
    return metadata_path


def _transfer_audio_directory(source: Path, destination: Path, mode: str) -> None:
    if mode == "move":
        source.rename(destination)
        return

    copy_function = os.link if mode == "hardlink" else shutil.copy2
    try:
        shutil.copytree(source, destination, copy_function=copy_function)
    except OSError as error:
        if mode == "hardlink":
            raise OSError(
                "Hard-link creation failed. Source and output must be on the same "
                "filesystem; use --mode copy otherwise."
            ) from error
        raise


def _restore_moved_audio(
    source: Path,
    temporary_destination: Path,
) -> None:
    (temporary_destination / "metadata.csv").unlink(missing_ok=True)
    (temporary_destination / "metadata.csv.tmp").unlink(missing_ok=True)
    if temporary_destination.exists() and not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        temporary_destination.rename(source)


def prepare_subset(prepared: PreparedSubset, mode: str) -> Path:
    """Transfer one audio directory and atomically publish its HF dataset."""

    if prepared.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {prepared.output_dir}. "
            "Choose another datasets directory or remove it explicitly."
        )
    prepared.output_dir.parent.mkdir(parents=True, exist_ok=True)
    if (
        mode in {"hardlink", "move"}
        and prepared.source_audio_dir.stat().st_dev
        != prepared.output_dir.parent.stat().st_dev
    ):
        raise OSError(
            f"--mode {mode} requires source and output on the same filesystem. "
            "Use --mode copy otherwise."
        )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{prepared.output_dir.name}.tmp-",
            dir=prepared.output_dir.parent,
        )
    )
    temporary_audio_dir = staging_dir / "data" / "test"
    temporary_audio_dir.parent.mkdir(parents=True)
    moved = False
    try:
        _transfer_audio_directory(
            prepared.source_audio_dir,
            temporary_audio_dir,
            mode,
        )
        moved = mode == "move"
        metadata_path = write_metadata_csv(
            temporary_audio_dir,
            prepared.records,
        )
        logger.info(
            "Prepared %s with %d trial rows",
            metadata_path,
            len(prepared.records),
        )
        staging_dir.rename(prepared.output_dir)
    except Exception:
        if moved:
            try:
                _restore_moved_audio(
                    prepared.source_audio_dir,
                    temporary_audio_dir,
                )
            except OSError as rollback_error:
                raise RuntimeError(
                    "Conversion failed and the moved audio directory could not be "
                    f"restored. Staging data remains at {staging_dir}."
                ) from rollback_error
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return prepared.output_dir


def prepare_hf_datasets(
    datasets_dir: Path,
    keys_dir: Path = DEFAULT_KEYS_DIR,
    subset: str = "both",
    mode: str = "hardlink",
    dry_run: bool = False,
) -> tuple[Path, ...]:
    """Validate and prepare one or both ASVspoof2021 evaluation datasets."""

    if subset not in {"DF", "LA", "both"}:
        raise ValueError("subset must be one of: DF, LA, both.")
    if mode not in {"hardlink", "move", "copy"}:
        raise ValueError("mode must be one of: hardlink, move, copy.")

    datasets_dir = datasets_dir.expanduser().resolve()
    keys_dir = keys_dir.expanduser().resolve()
    selected_names = ("DF", "LA") if subset == "both" else (subset,)
    prepared_subsets = tuple(
        inspect_subset(SUBSET_SPECS[name], datasets_dir, keys_dir)
        for name in selected_names
    )

    for prepared in prepared_subsets:
        logger.info(
            "%s: source=%s, output=%s, trial rows=%d, FLAC files=%d, "
            "unreferenced FLAC files=%d",
            prepared.spec.name,
            prepared.source_audio_dir,
            prepared.output_dir,
            len(prepared.records),
            prepared.audio_count,
            prepared.extra_audio_count,
        )
        if prepared.output_dir.exists():
            message = f"Output directory already exists: {prepared.output_dir}"
            if dry_run:
                logger.warning("%s", message)
            else:
                raise FileExistsError(message)

    if dry_run:
        return tuple(prepared.output_dir for prepared in prepared_subsets)

    return tuple(
        prepare_subset(prepared, mode)
        for prepared in prepared_subsets
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    outputs = prepare_hf_datasets(
        datasets_dir=args.datasets_dir,
        keys_dir=args.keys_dir,
        subset=args.subset,
        mode=args.mode,
        dry_run=args.dry_run,
    )
    for output in outputs:
        logger.info("Hugging Face dataset output: %s", output)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_DATASETS_DIR",
    "DEFAULT_KEYS_DIR",
    "METADATA_FIELDS",
    "PreparedSubset",
    "SUBSET_SPECS",
    "SubsetSpec",
    "TrialRecord",
    "inspect_subset",
    "prepare_hf_datasets",
    "prepare_subset",
    "read_trials",
    "write_metadata_csv",
]
