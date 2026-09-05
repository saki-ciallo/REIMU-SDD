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
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
DEFAULT_SOURCE_DIR = DATASETS_ROOT / "ASVspoof2019"
DEFAULT_OUTPUT_NAME = "ASVspoof2019_hf"


@dataclass(frozen=True)
class ProtocolRecord:
    speaker_id: str
    utterance_id: str
    system_id: str
    label: str

    @property
    def audio_filename(self) -> str:
        return f"{self.utterance_id}.flac"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    protocol_filename: str
    source_audio_dirname: str


@dataclass(frozen=True)
class PreparedSplit:
    spec: SplitSpec
    protocol_path: Path
    source_audio_dir: Path
    records: tuple[ProtocolRecord, ...]
    audio_count: int
    extra_audio_count: int


SPLIT_SPECS = (
    SplitSpec(
        name="train",
        protocol_filename="ASVspoof2019.LA.cm.train.trn.txt",
        source_audio_dirname="ASVspoof2019_LA_train",
    ),
    SplitSpec(
        name="validation",
        protocol_filename="ASVspoof2019.LA.cm.dev.trl.txt",
        source_audio_dirname="ASVspoof2019_LA_dev",
    ),
    SplitSpec(
        name="test",
        protocol_filename="ASVspoof2019.LA.cm.eval.trl.txt",
        source_audio_dirname="ASVspoof2019_LA_eval",
    ),
)

METADATA_FIELDS = (
    "file_name",
    "speaker_id",
    "utterance_id",
    "system_id",
    "labels",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rearrange the downloaded ASVspoof2019 LA release into a Hugging Face "
            "audiofolder dataset and generate metadata.csv for every split."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=(
            "Downloaded ASVspoof2019 root. The LA release may be nested as LA/LA. "
            f"Default: {DEFAULT_SOURCE_DIR}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output dataset directory. Defaults to an ASVspoof2019_hf sibling of "
            "--source-dir."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "move", "copy"),
        default="hardlink",
        help=(
            "Audio transfer method. hardlink preserves the source without duplicating "
            "audio data; move renames each flac directory; copy duplicates the files."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the source and print the planned conversion without changing files.",
    )
    return parser.parse_args(argv)


def _is_la_root(path: Path) -> bool:
    return (
        (path / "ASVspoof2019_LA_cm_protocols").is_dir()
        and (path / "ASVspoof2019_LA_train" / "flac").is_dir()
        and (path / "ASVspoof2019_LA_dev" / "flac").is_dir()
        and (path / "ASVspoof2019_LA_eval" / "flac").is_dir()
    )


def resolve_la_root(source_dir: Path) -> Path:
    """Locate the LA release while deliberately ignoring the PA branch."""

    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source_dir}")

    direct_candidates = (
        source_dir,
        source_dir / "LA",
        source_dir / "LA" / "LA",
    )
    for candidate in direct_candidates:
        if _is_la_root(candidate):
            return candidate

    protocol_dirs = [
        path
        for path in source_dir.rglob("ASVspoof2019_LA_cm_protocols")
        if path.is_dir() and "PA" not in path.parts and _is_la_root(path.parent)
    ]
    if len(protocol_dirs) == 1:
        return protocol_dirs[0].parent
    if not protocol_dirs:
        raise FileNotFoundError(
            "Could not locate ASVspoof2019 LA protocols and train/dev/eval flac "
            f"directories under {source_dir}."
        )
    matches = ", ".join(str(path.parent) for path in protocol_dirs)
    raise RuntimeError(f"Found multiple possible ASVspoof2019 LA roots: {matches}")


def read_protocol(protocol_path: Path) -> tuple[ProtocolRecord, ...]:
    """Parse the five-column ASVspoof2019 CM protocol."""

    records: list[ProtocolRecord] = []
    seen_utterance_ids: set[str] = set()
    with protocol_path.open(encoding="utf-8") as protocol_file:
        for line_number, raw_line in enumerate(protocol_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"{protocol_path}:{line_number} must contain 5 fields, "
                    f"got {len(fields)}."
                )

            speaker_id, utterance_id, _, system_id, label = fields
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
                ProtocolRecord(
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                    system_id=system_id,
                    label=label,
                )
            )

    if not records:
        raise ValueError(f"Protocol file is empty: {protocol_path}")
    return tuple(records)


def inspect_source_splits(la_root: Path) -> tuple[PreparedSplit, ...]:
    """Read all protocols and verify that every referenced FLAC exists."""

    protocol_dir = la_root / "ASVspoof2019_LA_cm_protocols"
    prepared_splits: list[PreparedSplit] = []
    for spec in SPLIT_SPECS:
        protocol_path = protocol_dir / spec.protocol_filename
        source_audio_dir = la_root / spec.source_audio_dirname / "flac"
        if not protocol_path.is_file():
            raise FileNotFoundError(f"Missing protocol file: {protocol_path}")
        if not source_audio_dir.is_dir():
            raise FileNotFoundError(f"Missing audio directory: {source_audio_dir}")

        records = read_protocol(protocol_path)
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
                f"{spec.name} is missing {len(missing_filenames)} protocol audio files. "
                f"Examples: {examples}"
            )

        prepared_splits.append(
            PreparedSplit(
                spec=spec,
                protocol_path=protocol_path,
                source_audio_dir=source_audio_dir,
                records=records,
                audio_count=len(audio_filenames),
                extra_audio_count=len(audio_filenames - expected_filenames),
            )
        )
    return tuple(prepared_splits)


def write_metadata_csv(
    output_split_dir: Path,
    records: Sequence[ProtocolRecord],
) -> Path:
    """Write Hugging Face audiofolder metadata next to the split audio files."""

    metadata_path = output_split_dir / "metadata.csv"
    temporary_path = output_split_dir / "metadata.csv.tmp"
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "file_name": record.audio_filename,
                    "speaker_id": record.speaker_id,
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


def _rollback_moved_directories(
    moved_directories: Sequence[tuple[Path, Path]],
) -> list[str]:
    rollback_errors: list[str] = []
    for source, temporary_destination in reversed(moved_directories):
        try:
            (temporary_destination / "metadata.csv").unlink(missing_ok=True)
            (temporary_destination / "metadata.csv.tmp").unlink(missing_ok=True)
            if temporary_destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                temporary_destination.rename(source)
        except OSError as error:
            rollback_errors.append(
                f"{temporary_destination} -> {source}: {error}"
            )
    return rollback_errors


def prepare_hf_dataset(
    source_dir: Path,
    output_dir: Path,
    mode: str = "hardlink",
    dry_run: bool = False,
) -> Path:
    """Validate, rearrange, and annotate the ASVspoof2019 LA dataset."""

    if mode not in {"move", "hardlink", "copy"}:
        raise ValueError("mode must be one of: move, hardlink, copy.")

    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    la_root = resolve_la_root(source_dir)
    prepared_splits = inspect_source_splits(la_root)

    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("output_dir must not be the source directory or inside it.")
    if output_dir.exists() and not dry_run:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose another path or remove it explicitly."
        )

    logger.info("ASVspoof2019 LA source: %s", la_root)
    logger.info("Hugging Face output: %s", output_dir)
    logger.info("Transfer mode: %s", mode)
    for prepared in prepared_splits:
        logger.info(
            "%s: %d protocol rows, %d FLAC files, %d unreferenced FLAC files",
            prepared.spec.name,
            len(prepared.records),
            prepared.audio_count,
            prepared.extra_audio_count,
        )
    if dry_run:
        if output_dir.exists():
            logger.warning("Output already exists and a real conversion would stop.")
        return output_dir

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        output_device = output_dir.parent.stat().st_dev
        for prepared in prepared_splits:
            if prepared.source_audio_dir.stat().st_dev != output_device:
                raise OSError(
                    "--mode move requires source and output on the same filesystem. "
                    "Use --mode copy for a cross-filesystem conversion."
                )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    moved_directories: list[tuple[Path, Path]] = []
    try:
        data_dir = staging_dir / "data"
        data_dir.mkdir()
        for prepared in prepared_splits:
            output_split_dir = data_dir / prepared.spec.name
            _transfer_audio_directory(
                prepared.source_audio_dir,
                output_split_dir,
                mode,
            )
            if mode == "move":
                moved_directories.append(
                    (prepared.source_audio_dir, output_split_dir)
                )
            metadata_path = write_metadata_csv(
                output_split_dir,
                prepared.records,
            )
            logger.info(
                "Prepared %s with %d metadata rows",
                metadata_path,
                len(prepared.records),
            )

        staging_dir.rename(output_dir)
    except Exception:
        rollback_errors = (
            _rollback_moved_directories(moved_directories)
            if mode == "move"
            else []
        )
        if rollback_errors:
            details = "\n".join(rollback_errors)
            raise RuntimeError(
                "Conversion failed and one or more moved directories could not be "
                f"restored. Staging data remains at {staging_dir}:\n{details}"
            )
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    logger.info("ASVspoof2019 Hugging Face dataset is ready: %s", output_dir)
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else source_dir.with_name(DEFAULT_OUTPUT_NAME)
    )
    prepare_hf_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        mode=args.mode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "DEFAULT_SOURCE_DIR",
    "METADATA_FIELDS",
    "PreparedSplit",
    "ProtocolRecord",
    "SPLIT_SPECS",
    "SplitSpec",
    "inspect_source_splits",
    "prepare_hf_dataset",
    "read_protocol",
    "resolve_la_root",
    "write_metadata_csv",
]
