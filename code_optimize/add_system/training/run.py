from __future__ import annotations

import logging
from time import perf_counter

import torch
from transformers import EarlyStoppingCallback, set_seed

from ..configuration.experiments import ResolvedExperiment
from ..data import AudioClassificationCollator, apply_rawboost_transforms, load_splits
from ..diagnostics import write_model_report, write_run_manifest
from ..models import ADDModel
from .arguments import build_training_arguments
from .losses import build_loss
from .trainer import ADDTrainer, compute_metrics

LOGGER = logging.getLogger(__name__)


def run_training(resolved: ResolvedExperiment) -> None:
    startup_started = perf_counter()
    settings = resolved.build_settings()
    if not torch.cuda.is_available():
        raise RuntimeError("Optimized ADD training requires CUDA.")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    resolved.save(settings.output_dir / "resolved_config.yaml")
    set_seed(settings.run.seed)

    model_started = perf_counter()
    training_arguments = build_training_arguments(settings)
    model = ADDModel(settings.model)
    model_seconds = perf_counter() - model_started

    report_started = perf_counter()
    report_path = write_model_report(model, settings.output_dir)
    report_seconds = perf_counter() - report_started

    data_started = perf_counter()
    train_dataset, eval_dataset = load_splits(
        settings.data,
        do_train=settings.training.do_train,
        do_eval=settings.training.do_eval,
    )
    train_dataset, eval_dataset = apply_rawboost_transforms(
        train_dataset,
        eval_dataset,
        augmentation=settings.augmentation,
        data=settings.data,
    )
    if settings.augmentation.enabled:
        LOGGER.info(
            "Original CPU RawBoost enabled: algorithm=%s, validation=%s",
            settings.augmentation.algorithm,
            settings.augmentation.apply_to_validation,
        )
    data_seconds = perf_counter() - data_started
    loss_function = build_loss(
        settings.loss,
        settings.model.classifier_config.num_labels,
    )
    callbacks = []
    if settings.training.do_eval and settings.training.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=settings.training.early_stopping_patience,
                early_stopping_threshold=settings.training.early_stopping_threshold,
            )
        )

    trainer_started = perf_counter()
    trainer = ADDTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AudioClassificationCollator(settings.data),
        compute_metrics=compute_metrics if settings.training.do_eval else None,
        callbacks=callbacks,
        loss_function=loss_function,
    )
    trainer_seconds = perf_counter() - trainer_started
    startup_timings = {
        "model_and_arguments": model_seconds,
        "model_report": report_seconds,
        "dataset_loading": data_seconds,
        "trainer_initialization": trainer_seconds,
        "total": perf_counter() - startup_started,
    }
    manifest_path = write_run_manifest(settings, resolved, startup_timings)

    LOGGER.info("Task: %s", settings.task_name)
    LOGGER.info("Output: %s", settings.output_dir)
    LOGGER.info("Model report: %s", report_path)
    LOGGER.info("Run manifest: %s", manifest_path)
    LOGGER.info(
        "GPU: %s; BF16=%s; compile=%s/%s",
        torch.cuda.get_device_name(0),
        settings.training.bf16,
        settings.training.torch_compile,
        settings.training.torch_compile_mode,
    )
    LOGGER.info(
        "Startup seconds: model=%.3f report=%.3f data=%.3f trainer=%.3f total=%.3f",
        model_seconds,
        report_seconds,
        data_seconds,
        trainer_seconds,
        startup_timings["total"],
    )
    if settings.training.do_train:
        result = trainer.train(resume_from_checkpoint=settings.training.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        LOGGER.info("Best checkpoint: %s", trainer.state.best_model_checkpoint)
    if settings.training.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
