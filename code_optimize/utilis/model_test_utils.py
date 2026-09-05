from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import TrainingArguments

from add_system.configuration.experiments import resolve_experiment
from add_system.configuration.settings import DataSettings, ExperimentSettings
from add_system.data import AudioClassificationCollator
from add_system.models import ADDModel
from add_system.training import ADDTrainer
from add_system.training.trainer import compute_metrics

from .model_resutls_utils import get_tested_results_dir, save_prediction_table

logger = logging.getLogger(__name__)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
OUTPUTS_ROOT = PACKAGE_ROOT / "outputs"
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


@dataclass(frozen=True)
class ModelVariant:
    name: str
    path: Path
    step: int | None

    @property
    def step_label(self) -> str:
        return str(self.step) if self.step is not None else self.name


def checkpoint_step(path: str | Path) -> int | None:
    match = CHECKPOINT_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def resolve_model_path(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == PACKAGE_ROOT.name:
        return (REPOSITORY_ROOT / path).resolve()
    if path.parts and path.parts[0] == OUTPUTS_ROOT.name:
        return (PACKAGE_ROOT / path).resolve()
    return (OUTPUTS_ROOT / path).resolve()


def load_trainer_state(model_dir: Path) -> dict[str, object]:
    state_path = model_dir / "trainer_state.json"
    if not state_path.is_file():
        return {}
    with state_path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise TypeError(f"Trainer state must be a JSON object: {state_path}")
    return state


def resolve_checkpoint_path(model_dir: Path, checkpoint: str | None) -> Path | None:
    if not checkpoint:
        return None

    checkpoint_path = Path(checkpoint)
    candidates: list[Path]
    if checkpoint_path.is_absolute():
        candidates = [checkpoint_path]
    else:
        candidates = [
            Path.cwd() / checkpoint_path,
            model_dir / checkpoint_path,
            model_dir / checkpoint_path.name,
        ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def find_last_checkpoint(model_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in model_dir.glob("checkpoint-*")
        if path.is_dir() and checkpoint_step(path) is not None
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: checkpoint_step(path) or -1).resolve()


def has_saved_model(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
    )


def discover_model_variants(
    model_path: str | Path,
    checkpoint_mode: str,
) -> list[ModelVariant]:
    if checkpoint_mode not in {"best", "last", "both"}:
        raise ValueError("checkpoint_mode must be 'best', 'last', or 'both'.")

    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"model_path does not exist or is not a directory: {path}")

    direct_step = checkpoint_step(path)
    if direct_step is not None:
        return [ModelVariant(name="checkpoint", path=path, step=direct_step)]

    trainer_state = load_trainer_state(path)
    variants: list[ModelVariant] = []

    if checkpoint_mode in {"best", "both"}:
        best_path = resolve_checkpoint_path(
            path,
            trainer_state.get("best_model_checkpoint"),
        )
        best_step = trainer_state.get("best_global_step")
        if not isinstance(best_step, int):
            best_step = checkpoint_step(best_path) if best_path is not None else None
        if best_path is not None:
            variants.append(ModelVariant(name="best", path=best_path, step=best_step))
        elif has_saved_model(path):
            variants.append(ModelVariant(name="best", path=path, step=best_step))

    if checkpoint_mode in {"last", "both"}:
        last_path = find_last_checkpoint(path)
        if last_path is not None:
            variants.append(
                ModelVariant(
                    name="last",
                    path=last_path,
                    step=checkpoint_step(last_path),
                )
            )
        elif checkpoint_mode == "last" and has_saved_model(path):
            variants.append(ModelVariant(name="root", path=path, step=None))

    deduped: list[ModelVariant] = []
    seen: set[Path] = set()
    for variant in variants:
        if variant.path in seen:
            continue
        seen.add(variant.path)
        deduped.append(variant)

    if not deduped:
        raise FileNotFoundError(
            f"No loadable model variant found under {path}. "
            "Expected root model files or checkpoint-* directories."
        )
    return deduped


def load_test_dataset(
    path: str | Path,
    split: str,
    max_samples: int | None,
) -> Dataset:
    dataset = load_from_disk(str(Path(path).expanduser()))
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(
                f"Split {split!r} not found in dataset at {path}. Available: {list(dataset)}"
            )
        dataset = dataset[split]
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Expected Dataset or DatasetDict from {path}, got {type(dataset).__name__}."
        )
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive or null.")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def make_test_training_args(
    output_dir: Path,
    settings: ExperimentSettings | None,
    per_device_eval_batch_size: int | None = None,
    dataloader_num_workers: int | None = None,
    bf16: bool | None = None,
) -> TrainingArguments:
    training = settings.training if settings is not None else None
    resolved_batch_size = (
        per_device_eval_batch_size
        if per_device_eval_batch_size is not None
        else (training.per_device_eval_batch_size if training is not None else 32)
    )
    resolved_workers = (
        dataloader_num_workers
        if dataloader_num_workers is not None
        else (training.dataloader_num_workers if training is not None else 0)
    )
    resolved_bf16 = bf16 if bf16 is not None else (training.bf16 if training is not None else True)
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=False,
        do_eval=False,
        per_device_eval_batch_size=resolved_batch_size,
        dataloader_num_workers=resolved_workers,
        dataloader_persistent_workers=resolved_workers > 0,
        bf16=resolved_bf16,
        bf16_full_eval=resolved_bf16,
        remove_unused_columns=(training.remove_unused_columns if training is not None else True),
        label_names=["labels"],
        report_to=[],
    )


def predict_one_dataset(
    model: ADDModel,
    training_args: TrainingArguments,
    dataset: Dataset,
    output_csv: Path,
    data_settings: DataSettings | None,
    loss_fn: torch.nn.Module,
    input_values_column_name: str = "input_values",
    label_column_name: str = "labels",
    metric_key_prefix: str | None = None,
) -> dict[str, float]:
    if data_settings is None:
        data_settings = DataSettings(
            dataset_key="unknown",
            dataset_cache_path="evaluation-only",
        )
    elif (
        data_settings.input_column != input_values_column_name
        or data_settings.label_column != label_column_name
    ):
        data_settings = replace(
            data_settings,
            input_column=input_values_column_name,
            label_column=label_column_name,
        )
    trainer = ADDTrainer(
        model=model,
        args=training_args,
        data_collator=AudioClassificationCollator(data_settings),
        compute_metrics=compute_metrics,
        loss_function=loss_fn,
    )
    metric_key_prefix = metric_key_prefix or output_csv.stem
    predict_result = trainer.predict(dataset, metric_key_prefix=metric_key_prefix)
    logits = np.asarray(predict_result.predictions)
    finite_mask = np.isfinite(logits)
    if not finite_mask.all():
        nonfinite_count = int(logits.size - finite_mask.sum())
        raise FloatingPointError(
            f"Prediction logits contain {nonfinite_count} non-finite values; "
            f"refusing to write {output_csv}."
        )

    logger.info("%s metrics: %s", metric_key_prefix, predict_result.metrics)
    metrics_path = output_csv.with_name(f"{output_csv.stem}_metrics.json")
    metrics_path.write_text(
        json.dumps(predict_result.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_prediction_table(
        predict_result=predict_result,
        dataset=dataset,
        output_csv=str(output_csv),
        id_column="utterance_id",
        label_column=data_settings.label_column,
    )
    return {key: float(value) for key, value in predict_result.metrics.items()}


def resolve_output_dir(model_path: Path) -> Path:
    run_dir = model_path.parent if checkpoint_step(model_path) is not None else model_path
    return get_tested_results_dir(run_dir).resolve()


def load_run_settings(output_dir: Path) -> ExperimentSettings | None:
    config_path = output_dir.parent / "resolved_config.yaml"
    if not config_path.is_file():
        logger.warning(
            "Missing %s; using default evaluation settings. Predictions and EER are unaffected.",
            config_path,
        )
        return None
    return resolve_experiment([config_path]).build_settings()


__all__ = [
    "ModelVariant",
    "checkpoint_step",
    "discover_model_variants",
    "load_run_settings",
    "load_test_dataset",
    "make_test_training_args",
    "predict_one_dataset",
    "resolve_model_path",
    "resolve_output_dir",
]
