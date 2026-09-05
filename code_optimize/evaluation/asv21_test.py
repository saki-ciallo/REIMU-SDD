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


def selected_dataset_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs = {
        "la": ("asv21la", Path(args.asv21_la_path).expanduser()),
        "df": ("asv21df", Path(args.asv21_df_path).expanduser()),
    }
    if args.datasets == "both":
        return [specs["la"], specs["df"]]
    return [specs[args.datasets]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ADDModel checkpoints on ASVspoof2021 LA/DF datasets."
    )
    parser.add_argument(
        "--model_path",
        "--model-dir",
        dest="model_path",
        required=True,
        help="Run name under outputs, or an absolute model/checkpoint path.",
    )
    parser.add_argument(
        "--asv21_la_path",
        default=str(DATASETS_ROOT / "ASVspoof2021_LA_16k_4s_fixed"),
    )
    parser.add_argument(
        "--asv21_df_path",
        default=str(DATASETS_ROOT / "ASVspoof2021_DF_16k_4s_fixed"),
    )
    parser.add_argument("--datasets", choices=["la", "df", "both"], default="both")
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
            "The ASV21 test CLI expects CUDA because the ADD/FLA model uses GPU kernels."
        )

    model_path = resolve_model_path(args.model_path)
    output_dir = resolve_output_dir(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_run_settings(output_dir)
    variants = discover_model_variants(model_path, args.checkpoint_mode)
    datasets = [
        (prefix, load_test_dataset(path, args.split, args.max_predict_samples))
        for prefix, path in selected_dataset_specs(args)
    ]
    training_args = make_test_training_args(
        output_dir=output_dir / "test_asv21_trainer",
        settings=settings,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
    )

    logger.info(
        "Model variants: %s",
        [(variant.name, str(variant.path), variant.step) for variant in variants],
    )
    logger.info("Output directory: %s", output_dir)
    logger.info(
        "Datasets: %s",
        [(prefix, len(dataset)) for prefix, dataset in datasets],
    )
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
        for prefix, dataset in datasets:
            output_csv = output_dir / f"{prefix}_{variant.step_label}.csv"
            logger.info(
                "Predicting %s with %s -> %s",
                prefix,
                variant.name,
                output_csv,
            )
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
