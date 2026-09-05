from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import TrainingArguments

from src.modeling_add import ADDModel
from utilis.config_utils import load_yaml_config
from utilis.model_resutls_utils import get_tested_results_dir, save_prediction_table
from utilis.trainer_utils import (
    ADDTrainer,
    AudioClassificationCollator,
    LossArguments,
    compute_metrics,
    preprocess_logits_for_metrics,
)


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
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
    if path.parts and path.parts[0] == OUTPUTS_ROOT.name:
        return (PROJECT_ROOT / path).resolve()
    return (OUTPUTS_ROOT / path).resolve()


def load_trainer_state(model_dir: Path) -> dict:
    state_path = model_dir / "trainer_state.json"
    if not state_path.is_file():
        return {}
    with state_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_checkpoint_path(model_dir: Path, checkpoint: str | None) -> Path | None:
    if not checkpoint:
        return None

    checkpoint_path = Path(checkpoint)
    candidates = []
    if checkpoint_path.is_absolute():
        candidates.append(checkpoint_path)
    else:
        candidates.extend(
            [
                Path.cwd() / checkpoint_path,
                model_dir / checkpoint_path,
                model_dir / checkpoint_path.name,
            ]
        )

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
        (path / "model.safetensors").is_file()
        or (path / "pytorch_model.bin").is_file()
    )


def discover_model_variants(
    model_path: str | Path,
    checkpoint_mode: str,
) -> list[ModelVariant]:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"model_path does not exist or is not a directory: {path}"
        )

    if checkpoint_step(path) is not None:
        return [
            ModelVariant(
                name="checkpoint",
                path=path,
                step=checkpoint_step(path),
            )
        ]

    trainer_state = load_trainer_state(path)
    variants: list[ModelVariant] = []

    if checkpoint_mode in {"best", "both"}:
        best_path = resolve_checkpoint_path(
            path,
            trainer_state.get("best_model_checkpoint"),
        )
        best_step = trainer_state.get("best_global_step")
        if best_step is None and best_path is not None:
            best_step = checkpoint_step(best_path)

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
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(
                f"Split {split!r} not found in dataset at {path}. "
                f"Available: {list(dataset)}"
            )
        dataset = dataset[split]
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Expected Dataset or DatasetDict from {path}, "
            f"got {type(dataset).__name__}."
        )
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def make_test_training_args(
    output_dir: Path,
    run_config: dict[str, object],
    per_device_eval_batch_size: int | None = None,
    dataloader_num_workers: int | None = None,
    bf16: bool | None = None,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=(
            per_device_eval_batch_size
            if per_device_eval_batch_size is not None
            else int(run_config.get("per_device_eval_batch_size", 32))
        ),
        dataloader_num_workers=(
            dataloader_num_workers
            if dataloader_num_workers is not None
            else int(run_config.get("dataloader_num_workers", 0))
        ),
        bf16=bf16 if bf16 is not None else bool(run_config.get("bf16", True)),
        remove_unused_columns=False,
        report_to=[],
    )


def predict_one_dataset(
    model: ADDModel,
    training_args: TrainingArguments,
    dataset: Dataset,
    output_csv: Path,
    input_values_column_name: str,
    label_column_name: str,
    loss_fn: torch.nn.Module,
    metric_key_prefix: str | None = None,
) -> dict[str, float]:
    trainer = ADDTrainer(
        model=model,
        args=training_args,
        data_collator=AudioClassificationCollator(
            input_values_column_name=input_values_column_name,
            label_column_name=label_column_name,
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        loss_fn=loss_fn,
    )
    metric_key_prefix = metric_key_prefix or output_csv.stem
    predict_result = trainer.predict(
        dataset,
        metric_key_prefix=metric_key_prefix,
    )
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
        label_column=label_column_name,
    )
    return predict_result.metrics


def resolve_output_dir(model_path: Path) -> Path:
    run_dir = model_path.parent if checkpoint_step(model_path) is not None else model_path
    return get_tested_results_dir(run_dir).resolve()


def load_run_config(output_dir: Path) -> tuple[dict[str, object], LossArguments]:
    config_path = output_dir.parent / "resolved_config.yaml"
    if config_path.is_file():
        run_config = load_yaml_config(config_path)
        return run_config, LossArguments.from_mapping(run_config)

    logger.warning(
        "Missing %s; using unweighted cross-entropy for test loss. "
        "Predictions and EER are unaffected.",
        config_path,
    )
    return {}, LossArguments(
        loss_type="cross_entropy",
        loss_bonafide_weight=1.0,
        loss_spoof_weight=1.0,
    )


__all__ = [
    "ModelVariant",
    "checkpoint_step",
    "discover_model_variants",
    "load_run_config",
    "load_test_dataset",
    "make_test_training_args",
    "predict_one_dataset",
    "resolve_model_path",
    "resolve_output_dir",
]
