from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

TESTED_RESULTS_DIRNAME = "tested_results"
SCORED_RESULTS_DIRNAME = "scored_results"

SCORE_SUMMARY_COLUMNS = [
    "dataset",
    "eer",
    "mtdcf",
    "epoch",
    "step",
    "loss",
    "accuracy",
]


def get_tested_results_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / TESTED_RESULTS_DIRNAME


def get_scored_results_dir(results_dir: str | Path) -> Path:
    return Path(results_dir) / SCORED_RESULTS_DIRNAME


def save_score_summary(
    scored_dir: str | Path,
    dataset: str,
    eer: float,
    mtdcf: float,
    epoch: float | None = None,
    step: int | None = None,
    loss: float | None = None,
    accuracy: float | None = None,
) -> Path:
    score_csv = Path(scored_dir) / "score.csv"
    row = {
        "dataset": dataset,
        "eer": eer,
        "mtdcf": mtdcf,
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "accuracy": accuracy,
    }

    if score_csv.is_file():
        df = pd.read_csv(score_csv)
        for column in SCORE_SUMMARY_COLUMNS:
            if column not in df.columns:
                df[column] = None
        df = df[df["dataset"] != dataset]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df = df.reindex(columns=SCORE_SUMMARY_COLUMNS)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(score_csv, index=False)
    return score_csv


def save_prediction_table(
    predict_result,
    dataset,
    output_csv: str,
    id_column: str = "utterance_id",
    label_column: str = "labels",
) -> None:
    logits = predict_result.predictions
    labels = predict_result.label_ids
    predictions = np.argmax(logits, axis=-1)

    data = {
        id_column: dataset[id_column],
        "label": labels,
        "prediction": predictions,
        "correct": predictions == labels,
    }

    if logits.ndim == 2 and logits.shape[1] == 2:
        data["bonafide_logit"] = logits[:, 0]
        data["spoof_logit"] = logits[:, 1]
        data["bonafide_score"] = logits[:, 0] - logits[:, 1]
        data["spoof_score"] = logits[:, 1] - logits[:, 0]

    # label_feature = dataset.features.get(label_column)
    # if hasattr(label_feature, "int2str"):
    #     data["label_name"] = [label_feature.int2str(int(x)) for x in labels]
    #     data["prediction_name"] = [
    #         label_feature.int2str(int(x)) for x in predictions
    #     ]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
