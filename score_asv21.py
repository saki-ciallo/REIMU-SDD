from __future__ import annotations

import argparse
import importlib.util
import io
import json
import logging
import re
import subprocess
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from utilis.model_resutls_utils import (
    get_scored_results_dir,
    get_tested_results_dir,
    save_score_summary,
)


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_EVAL_PACKAGE_DIR = PROJECT_ROOT / "official_scores" / "2021" / "eval-package"
STEP_RE = re.compile(r"_(\d+)$")
TRACK_PREFIXES = {
    "la": ("asv21la_", "LA"),
    "df": ("asv21df_", "DF"),
}


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
    if path.is_dir() and any(path.glob("asv21*.csv")):
        return path

    raise FileNotFoundError(
        f"Could not find ASV21 prediction results in {path} or {tested_results}."
    )


def infer_track(prediction_csv: str | Path) -> str:
    name = Path(prediction_csv).name.lower()
    for prefix, track in TRACK_PREFIXES.values():
        if name.startswith(prefix):
            return track
    raise ValueError(
        f"Cannot infer ASVspoof2021 track from {Path(prediction_csv).name!r}. "
        "Expected a file name beginning with 'asv21la_' or 'asv21df_'."
    )


def prediction_step(prediction_csv: str | Path) -> int | None:
    match = STEP_RE.search(Path(prediction_csv).stem)
    return int(match.group(1)) if match else None


def discover_prediction_files(
    results_dir: Path,
    datasets: str,
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
                raise FileNotFoundError(f"Missing ASV21 prediction file: {path}")
            infer_track(path)
            predictions.append(path)
    else:
        dataset_keys = ["la", "df"] if datasets == "both" else [datasets]
        predictions = []
        for dataset_key in dataset_keys:
            prefix, _ = TRACK_PREFIXES[dataset_key]
            predictions.extend(results_dir.glob(f"{prefix}*.csv"))

    deduped = sorted(
        set(predictions),
        key=lambda path: (infer_track(path), prediction_step(path) or -1, path.name),
    )
    if not deduped:
        raise FileNotFoundError(
            f"No ASV21 prediction csv files selected under {results_dir}."
        )
    return deduped


def write_asv21_cm_score(prediction_csv: Path, score_path: Path) -> Path:
    df = pd.read_csv(prediction_csv)
    required_columns = {"utterance_id", "bonafide_score"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{prediction_csv} is missing required columns: {sorted(missing_columns)}")

    utterance_ids = [str(value) for value in df["utterance_id"].tolist()]
    if len(utterance_ids) != len(set(utterance_ids)):
        raise ValueError(f"{prediction_csv} contains duplicate utterance_id values.")

    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", encoding="utf-8") as handle:
        for utterance_id, score in zip(utterance_ids, df["bonafide_score"].tolist(), strict=True):
            handle.write(f"{utterance_id} {float(score)}\n")
    return score_path


def _parse_float_token(token: str) -> float | None:
    token = token.strip().strip("\\")
    try:
        return float(token)
    except ValueError:
        return None


def _parse_pooled_value(table_text: str) -> float:
    for line in reversed(table_text.splitlines()):
        tokens = line.replace("&", " ").split()
        if not tokens or tokens[0] != "Pooled":
            continue
        values = [
            value
            for value in (_parse_float_token(token) for token in tokens[1:])
            if value is not None
        ]
        if values:
            return values[-1]
    raise ValueError("Could not parse pooled value from ASVspoof2021 official output.")


def parse_asv21_official_output(output_text: str, track: str) -> tuple[float, float]:
    eer_marker = "Table for EERs"
    if eer_marker not in output_text:
        raise ValueError("Could not find EER table in ASVspoof2021 official output.")

    if track == "DF":
        pooled_mtdcf = float("nan")
        eer_section = output_text.split(eer_marker, maxsplit=1)[1]
    else:
        mtdcf_marker = "Table for min tDCFs"
        if mtdcf_marker not in output_text:
            raise ValueError("Could not find min tDCF table in ASVspoof2021 official output.")
        mtdcf_section = output_text.split(mtdcf_marker, maxsplit=1)[1].split(
            eer_marker,
            maxsplit=1,
        )[0]
        eer_section = output_text.split(eer_marker, maxsplit=1)[1]
        pooled_mtdcf = _parse_pooled_value(mtdcf_section)

    pooled_eer_percent = _parse_pooled_value(eer_section)
    return pooled_eer_percent, pooled_mtdcf


def load_asv21_eval_module(eval_package_dir: Path):
    main_path = eval_package_dir / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(f"Missing ASVspoof2021 eval-package main.py: {main_path}")

    eval_package_str = str(eval_package_dir)
    if eval_package_str not in sys.path:
        sys.path.insert(0, eval_package_str)

    # Official metric computation does not depend on its Matplotlib-based table renderer.
    table_api_stub = types.ModuleType("table_API")

    def print_table(data_array, column_tag, row_tag, print_format="1.2f", **kwargs):
        lines = [" ".join([""] + [str(tag) for tag in column_tag])]
        for tag, values in zip(row_tag, data_array, strict=True):
            row = [str(tag)] + [f"{float(value):{print_format}}" for value in values]
            lines.append(" ".join(row))
        text_table = "\n".join(lines) + "\n"
        print(text_table)
        return "", text_table, [], []

    table_api_stub.print_table = print_table
    previous_table_api = sys.modules.get("table_API")
    sys.modules["table_API"] = table_api_stub

    spec = importlib.util.spec_from_file_location("_asv21_eval_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ASVspoof2021 eval package from {main_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_table_api is None:
            sys.modules.pop("table_API", None)
        else:
            sys.modules["table_API"] = previous_table_api
    return module


def run_asv21_official_main(
    score_path: Path,
    track: str,
    subset: str,
    eval_package_dir: Path,
) -> tuple[float, float, str]:
    command = [
        sys.executable,
        "main.py",
        "--cm-score-file",
        str(score_path),
        "--track",
        track,
        "--subset",
        subset,
    ]
    result = subprocess.run(
        command,
        cwd=eval_package_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    command_text = " ".join(command)
    output_text = (
        f"$ {command_text}\n\n"
        f"[stdout]\n{result.stdout}\n"
        f"[stderr]\n{result.stderr}\n"
    )
    if result.returncode != 0:
        return run_asv21_official_api_fallback(
            score_path,
            track,
            subset,
            eval_package_dir,
            output_text,
        )

    try:
        pooled_eer_percent, pooled_mtdcf = parse_asv21_official_output(result.stdout, track)
    except ValueError as exc:
        failed_output = f"{output_text}\n[parse error]\n{exc}\n"
        return run_asv21_official_api_fallback(
            score_path,
            track,
            subset,
            eval_package_dir,
            failed_output,
        )
    return pooled_eer_percent, pooled_mtdcf, output_text


def run_asv21_official_api_fallback(
    score_path: Path,
    track: str,
    subset: str,
    eval_package_dir: Path,
    failed_main_output: str,
) -> tuple[float, float, str]:
    eval_module = load_asv21_eval_module(eval_package_dir)
    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            score_result = eval_module.evaluation_API(
                str(score_path),
                track,
                subset,
                str(eval_package_dir / "keys"),
                False,
                None,
                None,
            )
    except SystemExit as exc:
        fallback_output = output_buffer.getvalue()
        raise RuntimeError(
            f"{failed_main_output}\n[fallback stdout]\n{fallback_output}\n"
            f"ASVspoof2021 official API fallback exited with {exc.code}."
        ) from exc
    except Exception as exc:
        fallback_output = output_buffer.getvalue()
        raise RuntimeError(
            f"{failed_main_output}\n[fallback stdout]\n{fallback_output}\n"
            f"ASVspoof2021 official API fallback failed: {exc}"
        ) from exc

    if score_result is None:
        raise RuntimeError(f"{failed_main_output}\nASVspoof2021 official API fallback returned no result.")

    mintdcf_array, eer_array = score_result
    pooled_mtdcf = float("nan") if track == "DF" else float(mintdcf_array[-1, -1])
    pooled_eer_percent = float(eer_array[-1, -1] * 100)
    output_text = (
        "[compatibility]\n"
        "The official CLI did not produce parseable tables in the current environment.\n"
        "Metrics below were computed by the official evaluation_API with plain-text rendering.\n\n"
        f"[stdout]\n{output_buffer.getvalue()}\n"
    )
    return pooled_eer_percent, pooled_mtdcf, output_text


def load_prediction_metrics(prediction_csv: Path) -> tuple[float | None, float | None]:
    metrics_path = prediction_csv.with_name(f"{prediction_csv.stem}_metrics.json")
    metrics = {}
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)

    loss = metrics.get(f"{prediction_csv.stem}_loss")
    accuracy = metrics.get(f"{prediction_csv.stem}_accuracy")
    if accuracy is None:
        prediction_df = pd.read_csv(prediction_csv, usecols=lambda column: column == "correct")
        if "correct" in prediction_df:
            accuracy = float(prediction_df["correct"].mean())
    return loss, accuracy


def score_prediction_file(
    prediction_csv: Path,
    eval_package_dir: Path,
    scored_dir: Path,
    subset: str,
) -> Path:
    track = infer_track(prediction_csv)
    score_path = scored_dir / f"{prediction_csv.stem}_cm_scores.txt"
    official_output_path = scored_dir / f"{prediction_csv.stem}_official_eval.txt"

    write_asv21_cm_score(prediction_csv, score_path)
    try:
        pooled_eer_percent, pooled_mtdcf, official_output = run_asv21_official_main(
            score_path=score_path,
            track=track,
            subset=subset,
            eval_package_dir=eval_package_dir,
        )
    except RuntimeError as exc:
        official_output_path.write_text(str(exc), encoding="utf-8")
        raise RuntimeError(
            f"ASVspoof2021 official scoring failed for {prediction_csv}. "
            f"See {official_output_path}."
        ) from exc

    official_output_path.write_text(official_output, encoding="utf-8")
    loss, accuracy = load_prediction_metrics(prediction_csv)
    score_csv = save_score_summary(
        scored_dir=scored_dir,
        dataset=prediction_csv.stem,
        eer=pooled_eer_percent,
        mtdcf=pooled_mtdcf,
        step=prediction_step(prediction_csv),
        loss=loss,
        accuracy=accuracy,
    )
    logger.info(
        "Scored %s (%s): pooled EER %.4f, min-tDCF %.6f",
        prediction_csv.name,
        track,
        pooled_eer_percent,
        pooled_mtdcf,
    )
    return score_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score existing ASVspoof2021 LA/DF prediction csv files."
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
        help="Prediction csv to score. Repeat this option for multiple files; omit it to scan the directory.",
    )
    parser.add_argument("--datasets", choices=["la", "df", "both"], default="both")
    parser.add_argument("--subset", default="eval")
    parser.add_argument("--eval-package-dir", default=str(DEFAULT_EVAL_PACKAGE_DIR))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir)
    eval_package_dir = Path(args.eval_package_dir).expanduser().resolve()
    scored_dir = get_scored_results_dir(results_dir)
    scored_dir.mkdir(parents=True, exist_ok=True)
    prediction_files = discover_prediction_files(results_dir, args.datasets, args.file_name)

    logger.info("Prediction directory: %s", results_dir)
    logger.info("Official eval package: %s", eval_package_dir)
    logger.info("Selected prediction files: %s", [path.name for path in prediction_files])

    score_csv = None
    for prediction_csv in prediction_files:
        score_csv = score_prediction_file(
            prediction_csv=prediction_csv,
            eval_package_dir=eval_package_dir,
            scored_dir=scored_dir,
            subset=args.subset,
        )
    logger.info("Score summary: %s", score_csv)


if __name__ == "__main__":
    main()
