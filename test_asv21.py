from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.modeling_add import ADDModel
from src.registration import register_add_for_auto
from utilis.model_test_utils import (
    discover_model_variants,
    load_run_config,
    load_test_dataset,
    make_test_training_args,
    predict_one_dataset,
    resolve_model_path,
    resolve_output_dir,
)
from utilis.trainer_utils import build_loss_fn


logger = logging.getLogger(__name__)


def selected_dataset_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs = {
        "la": ("asv21la", Path(args.asv21_la_path).expanduser()),
        "df": ("asv21df", Path(args.asv21_df_path).expanduser()),
    }
    if args.datasets == "both":
        return [specs["la"], specs["df"]]
    return [specs[args.datasets]]


def parse_args() -> argparse.Namespace:
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
        default="../datasets/ASVspoof2021_LA_16k_4s_fixed",
    )
    parser.add_argument(
        "--asv21_df_path",
        default="../datasets/ASVspoof2021_DF_16k_4s_fixed",
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "test_asv21.py expects CUDA because the ADD/FLA model uses GPU kernels."
        )

    register_add_for_auto()
    model_path = resolve_model_path(args.model_path)
    output_dir = resolve_output_dir(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config, loss_args = load_run_config(output_dir)
    variants = discover_model_variants(model_path, args.checkpoint_mode)
    datasets = [
        (prefix, load_test_dataset(path, args.split, args.max_predict_samples))
        for prefix, path in selected_dataset_specs(args)
    ]
    training_args = make_test_training_args(
        output_dir=output_dir / "test_asv21_trainer",
        run_config=run_config,
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
        loss_args.loss_type,
        loss_args.class_weights,
    )

    for variant in variants:
        logger.info("Loading %s checkpoint from %s", variant.name, variant.path)
        model = ADDModel.from_pretrained(str(variant.path)).cuda().eval()
        loss_fn = build_loss_fn(
            loss_args,
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
                input_values_column_name=args.input_values_column_name,
                label_column_name=args.label_column_name,
                loss_fn=loss_fn,
            )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
