from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from add_system.configuration.settings import LossSettings
from add_system.models import ADDModel
from add_system.training.losses import build_loss
from utilis.model_test_utils import (
    discover_model_variants,
    load_run_settings,
    load_test_dataset,
    make_test_training_args,
    predict_one_dataset,
    resolve_model_path,
    resolve_output_dir,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ADDModel checkpoints on the ASVspoof2019 test split."
    )
    parser.add_argument(
        "--model_path",
        "--model-dir",
        dest="model_path",
        required=True,
        help="Run name under outputs, or an absolute model/checkpoint path.",
    )
    parser.add_argument(
        "--dataset_path",
        default=str(DATASETS_ROOT / "ASVspoof2019_16k_4s_fixed"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--checkpoint_mode",
        choices=["best", "last", "both"],
        default="both",
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--max_predict_samples", type=int, default=None)
    parser.add_argument("--input_values_column_name", default="input_values")
    parser.add_argument("--label_column_name", default="labels")
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The ASV19 test CLI expects CUDA because the ADD/FLA model uses GPU kernels."
        )

    model_path = resolve_model_path(args.model_path)
    output_dir = resolve_output_dir(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_run_settings(output_dir)
    variants = discover_model_variants(model_path, args.checkpoint_mode)
    dataset = load_test_dataset(
        args.dataset_path,
        args.split,
        args.max_predict_samples,
    )
    training_args = make_test_training_args(
        output_dir=output_dir / "test_asv19_trainer",
        settings=settings,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
    )

    logger.info(
        "Model variants: %s",
        [(variant.name, str(variant.path), variant.step) for variant in variants],
    )
    logger.info("Dataset: %s split=%s samples=%s", args.dataset_path, args.split, len(dataset))
    logger.info("Output directory: %s", output_dir)
    logger.info(
        "Test loss: type=%s, class_weights=%s",
        settings.loss.loss_type if settings is not None else "cross_entropy",
        settings.loss.class_weights if settings is not None else LossSettings().class_weights,
    )

    for variant in variants:
        logger.info("Loading %s checkpoint from %s", variant.name, variant.path)
        model = ADDModel.from_pretrained(str(variant.path)).cuda().eval()
        loss_fn = build_loss(
            settings.loss if settings is not None else LossSettings(),
            model.config.classifier_config.num_labels,
        )
        output_csv = output_dir / f"asv19la_{variant.step_label}.csv"
        logger.info("Predicting ASV19 with %s -> %s", variant.name, output_csv)
        predict_one_dataset(
            model=model,
            training_args=training_args,
            dataset=dataset,
            output_csv=output_csv,
            data_settings=settings.data if settings is not None else None,
            input_values_column_name=args.input_values_column_name,
            label_column_name=args.label_column_name,
            loss_fn=loss_fn,
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
