from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utilis.get_asv19_scores import attack_level_eerandtdcf, eerandtdcf
from utilis.model_resutls_utils import (
    get_scored_results_dir,
    get_tested_results_dir,
    save_score_summary,
)


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
PREDICTION_PREFIX = "asv19la_"
STEP_RE = re.compile(r"_(\d+)$")


def resolve_results_dir(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        path = path.resolve()
    elif path.parts and path.parts[0] == OUTPUTS_ROOT.name:
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = (OUTPUTS_ROOT / path).resolve()

    if path.name == "tested_results" and path.is_dir():
        return path
    tested_results = get_tested_results_dir(path)
    if tested_results.is_dir():
        return tested_results.resolve()
    if path.is_dir() and any(path.glob(f"{PREDICTION_PREFIX}*.csv")):
        return path
    raise FileNotFoundError(
        f"Could not find ASV19 prediction results in {path} or {tested_results}."
    )


def prediction_step(prediction_csv: str | Path) -> int | None:
    match = STEP_RE.search(Path(prediction_csv).stem)
    return int(match.group(1)) if match else None


def discover_prediction_files(
    results_dir: Path,
    file_names: list[str] | None,
) -> list[Path]:
    if file_names:
        predictions = []
        for file_name in file_names:
            path = Path(file_name).expanduser()
            if not path.is_absolute():
                path = results_dir / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing ASV19 prediction file: {path}")
            if not path.name.lower().startswith(PREDICTION_PREFIX):
                raise ValueError(
                    f"ASV19 prediction file must start with {PREDICTION_PREFIX!r}: "
                    f"{path.name}"
                )
            predictions.append(path)
    else:
        predictions = list(results_dir.glob(f"{PREDICTION_PREFIX}*.csv"))

    predictions = sorted(
        set(predictions),
        key=lambda path: (prediction_step(path) or -1, path.name),
    )
    if not predictions:
        raise FileNotFoundError(
            f"No ASV19 prediction csv files selected under {results_dir}."
        )
    return predictions


def write_asv19_cm_score(prediction_csv: Path, score_path: Path) -> Path:
    df = pd.read_csv(prediction_csv)
    required_columns = {"utterance_id", "label", "bonafide_score"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{prediction_csv} is missing required columns: {sorted(missing_columns)}"
        )
    if df["utterance_id"].duplicated().any():
        raise ValueError(f"{prediction_csv} contains duplicate utterance_id values.")

    scores = df["bonafide_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError(
            f"{prediction_csv} contains non-finite bonafide_score values."
        )

    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", encoding="utf-8") as handle:
        for row in df.itertuples(index=False):
            label_name = "bonafide" if int(row.label) == 0 else "spoof"
            handle.write(
                f"{row.utterance_id} - {label_name} {float(row.bonafide_score)}\n"
            )
    return score_path


def load_prediction_metrics(
    prediction_csv: Path,
) -> tuple[float | None, float | None]:
    metrics_path = prediction_csv.with_name(
        f"{prediction_csv.stem}_metrics.json"
    )
    metrics = {}
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)

    loss = metrics.get(f"{prediction_csv.stem}_loss")
    accuracy = metrics.get(f"{prediction_csv.stem}_accuracy")
    if accuracy is None:
        prediction_df = pd.read_csv(
            prediction_csv,
            usecols=lambda column: column == "correct",
        )
        if "correct" in prediction_df:
            accuracy = float(prediction_df["correct"].mean())
    return loss, accuracy


def score_prediction_file(
    prediction_csv: Path,
    scored_dir: Path,
) -> Path:
    score_path = scored_dir / f"{prediction_csv.stem}_cm_scores.txt"
    attack_score_path = scored_dir / f"{prediction_csv.stem}_attack_scores.csv"
    write_asv19_cm_score(prediction_csv, score_path)

    eer, min_tdcf = eerandtdcf(score_path)
    attack_rows = attack_level_eerandtdcf(score_path)
    pd.DataFrame(
        attack_rows,
        columns=["attack", "bonafide_trials", "spoof_trials", "eer", "mtdcf"],
    ).to_csv(attack_score_path, index=False)

    loss, accuracy = load_prediction_metrics(prediction_csv)
    score_csv = save_score_summary(
        scored_dir=scored_dir,
        dataset=prediction_csv.stem,
        eer=eer,
        mtdcf=min_tdcf,
        step=prediction_step(prediction_csv),
        loss=loss,
        accuracy=accuracy,
    )
    logger.info(
        "Scored %s: EER %.4f, min-tDCF %.6f",
        prediction_csv.name,
        eer,
        min_tdcf,
    )
    logger.info("CM scores: %s", score_path)
    logger.info("Attack scores: %s", attack_score_path)
    return score_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score existing ASVspoof2019 LA prediction csv files."
    )
    parser.add_argument(
        "--results-dir",
        "--model-dir",
        dest="results_dir",
        required=True,
        help="Run name under outputs, or an absolute run/tested_results path.",
    )
    parser.add_argument(
        "--file-name",
        action="append",
        default=None,
        help=(
            "Prediction csv to score. Repeat for multiple files; "
            "omit it to scan all asv19la_*.csv files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir)
    scored_dir = get_scored_results_dir(results_dir)
    scored_dir.mkdir(parents=True, exist_ok=True)
    prediction_files = discover_prediction_files(results_dir, args.file_name)

    logger.info("Prediction directory: %s", results_dir)
    logger.info("Selected prediction files: %s", [path.name for path in prediction_files])
    score_csv = None
    for prediction_csv in prediction_files:
        score_csv = score_prediction_file(prediction_csv, scored_dir)
    logger.info("Score summary: %s", score_csv)


if __name__ == "__main__":
    main()
