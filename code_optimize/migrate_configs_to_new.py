from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml

from add_system.configuration.experiments import ConfigMapping, deep_merge, load_yaml_mapping

CODE_OPTIMIZE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODE_OPTIMIZE_ROOT.parent

# Keep the legacy schema explicit so a newly added source field cannot be
# silently dropped by the conversion below.
_LEGACY_CONFIG_FIELDS = frozenset(
    {
        "aasist_filts",
        "aasist_gat_dims",
        "aasist_graph_attention_dropout",
        "aasist_graph_pool_dropout",
        "aasist_input_size",
        "aasist_output_dropout",
        "aasist_path_dropout",
        "aasist_pool_ratios",
        "aasist_temperatures",
        "adam_beta1",
        "adam_beta2",
        "am_eps",
        "am_margin",
        "am_scale",
        "architecture_type",
        "attention_max_position_embeddings",
        "attention_num_heads",
        "attention_num_kv_heads",
        "attention_qk_norm",
        "attention_rope_theta",
        "attention_window_size",
        "bf16",
        "blocks_json",
        "ce_label_smoothing",
        "classifier_type",
        "cnn_channels",
        "cnn_kernels",
        "dataloader_num_workers",
        "dataset_cache_path",
        "dataset_key",
        "do_eval",
        "do_train",
        "early_stopping_patience",
        "early_stopping_threshold",
        "eval_split_name",
        "eval_strategy",
        "focal_gamma",
        "frame_ms",
        "freeze_ssl_model",
        "frontend_type",
        "gdn2_head_dim",
        "gdn2_num_heads",
        "gdn2_num_v_heads",
        "gpu_id",
        "greater_is_better",
        "hidden_size",
        "hrm_h_blocks_json",
        "hrm_h_cycles",
        "hrm_l_blocks_json",
        "hrm_l_cycles",
        "hrm_num_gradient_steps",
        "input_values_column_name",
        "label_column_name",
        "learning_rate",
        "load_best_model_at_end",
        "logging_steps",
        "logging_strategy",
        "looped_num_cycles",
        "looped_num_gradient_cycles",
        "loss_bonafide_weight",
        "loss_spoof_weight",
        "loss_type",
        "lr_scheduler_type",
        "mamba3_head_dim",
        "mamba3_is_mimo",
        "mamba3_rescale_prenorm_residual",
        "mamba3_state_size",
        "metric_for_best_model",
        "mlp_intermediate_size",
        "model_architecture",
        "moe_aux_loss_coeff",
        "moe_use_seq_aux_loss",
        "num_filters",
        "num_labels",
        "num_train_epochs",
        "optim",
        "output_root",
        "per_device_eval_batch_size",
        "per_device_train_batch_size",
        "pooling_dropout",
        "pooling_gate_act",
        "pooling_hidden_ratio",
        "pooling_num_heads",
        "pooling_output_act",
        "pooling_output_size",
        "pooling_type",
        "pooling_use_output_norm",
        "raven_decay_type",
        "raven_feature_map",
        "raven_num_heads",
        "raven_num_kv_heads",
        "raven_num_slots",
        "raven_router_score",
        "raven_router_type",
        "raven_topk",
        "rawboost_algo",
        "rawboost_apply_to_validation",
        "rawboost_impulse_gain",
        "rawboost_impulse_percent",
        "rawboost_max_bandwidth",
        "rawboost_max_coefficients",
        "rawboost_max_frequency",
        "rawboost_max_gain",
        "rawboost_max_nonlinear_bias",
        "rawboost_max_snr",
        "rawboost_min_bandwidth",
        "rawboost_min_coefficients",
        "rawboost_min_frequency",
        "rawboost_min_gain",
        "rawboost_min_nonlinear_bias",
        "rawboost_min_snr",
        "rawboost_nonlinearity_order",
        "rawboost_num_bands",
        "remove_unused_columns",
        "report_to",
        "residual_dropout",
        "return_aux_loss",
        "save_strategy",
        "save_total_limit",
        "seed",
        "sincnet_final_dropout",
        "ssl_trainable_layer_indices",
        "stride_ms",
        "target_sampling_rate",
        "task_name",
        "torch_compile",
        "torch_compile_mode",
        "train_split_name",
        "warmup_steps",
        "weight_decay",
        "window_ms",
    }
)


def _validate_legacy_fields(
    values: Mapping[str, object],
    *,
    source_name: str | Path = "Legacy configuration",
) -> None:
    unknown = set(values) - _LEGACY_CONFIG_FIELDS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"{source_name} contains unmapped legacy fields: {names}.")


def _blocks(value: object, *, field_name: str) -> object:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} is not valid JSON: {exc}") from exc
    if not isinstance(value, (Mapping, list)):
        raise TypeError(f"{field_name} must be a JSON mapping or list.")
    if isinstance(value, Mapping):
        return [deepcopy(item) for _, item in sorted(value.items(), key=lambda item: int(item[0]))]
    return deepcopy(value)


def _frontend(values: Mapping[str, object]) -> ConfigMapping:
    frontend_type = str(values["frontend_type"])
    pipeline = str(values["model_architecture"])
    output_size = (
        int(values["aasist_input_size"]) if pipeline == "ssl_aasist" else int(values["hidden_size"])
    )
    if frontend_type == "linear":
        return {
            "model_type": "linear_frontend",
            "target_sampling_rate": values["target_sampling_rate"],
            "frame_ms": values["frame_ms"],
            "output_size": output_size,
        }
    if frontend_type == "sincnet":
        channels = list(values["cnn_channels"])
        if not channels:
            raise ValueError("cnn_channels must not be empty.")
        channels[-1] = output_size
        return {
            "model_type": "sincnet_frontend",
            "target_sampling_rate": values["target_sampling_rate"],
            "window_ms": values["window_ms"],
            "stride_ms": values["stride_ms"],
            "num_filters": values["num_filters"],
            "cnn_kernels": deepcopy(values["cnn_kernels"]),
            "cnn_channels": channels,
            "final_dropout": values["sincnet_final_dropout"],
        }
    if frontend_type not in {
        "wav2vec2-base",
        "hubert-base-ls960",
        "wavlm-base",
        "wavlm-base-plus",
    }:
        raise ValueError(f"Unsupported frontend_type={frontend_type!r}.")
    return {
        "model_type": "ssl_frontend",
        "ssl_type": frontend_type,
        "target_sampling_rate": values["target_sampling_rate"],
        "output_size": output_size,
        "freeze_ssl_model": values["freeze_ssl_model"],
        "ssl_trainable_layer_indices": deepcopy(values["ssl_trainable_layer_indices"]),
        "local_files_only": False,
    }


def _classifier(values: Mapping[str, object], input_size: int) -> ConfigMapping:
    return {
        "model_type": "linear_classifier",
        "input_size": input_size,
        "num_labels": values["num_labels"],
        "classifier_type": values["classifier_type"],
        "am_margin": values["am_margin"],
        "am_scale": values["am_scale"],
        "am_eps": values["am_eps"],
    }


def _backbone(values: Mapping[str, object]) -> ConfigMapping:
    heterogeneous = values["architecture_type"] == "heterogeneous_hrm"
    blocks = None if heterogeneous else _blocks(values["blocks_json"], field_name="blocks_json")
    h_blocks = (
        _blocks(values["hrm_h_blocks_json"], field_name="hrm_h_blocks_json")
        if values.get("hrm_h_blocks_json") is not None
        else None
    )
    l_blocks = (
        _blocks(values["hrm_l_blocks_json"], field_name="hrm_l_blocks_json")
        if values.get("hrm_l_blocks_json") is not None
        else None
    )
    return {
        "model_type": "attention_backbone",
        "hidden_size": values["hidden_size"],
        "architecture_type": values["architecture_type"],
        "blocks": blocks,
        "hrm_h_blocks": h_blocks,
        "hrm_l_blocks": l_blocks,
        "looped_num_cycles": values["looped_num_cycles"],
        "looped_num_gradient_cycles": values["looped_num_gradient_cycles"],
        "hrm_h_cycles": values["hrm_h_cycles"],
        "hrm_l_cycles": values["hrm_l_cycles"],
        "hrm_num_gradient_steps": values["hrm_num_gradient_steps"],
        "mlp_intermediate_size": values["mlp_intermediate_size"],
        "residual_dropout": values["residual_dropout"],
        "moe_aux_loss_coeff": values["moe_aux_loss_coeff"],
        "moe_use_seq_aux_loss": values["moe_use_seq_aux_loss"],
        "return_aux_loss": values["return_aux_loss"],
        "use_cache": False,
        "attention_config": {
            "model_type": "attention_fla",
            "num_heads": values["attention_num_heads"],
            "num_kv_heads": values["attention_num_kv_heads"],
            "qk_norm": values["attention_qk_norm"],
            "window_size": values["attention_window_size"],
            "rope_theta": values["attention_rope_theta"],
            "max_position_embeddings": values["attention_max_position_embeddings"],
        },
        "raven_config": {
            "model_type": "raven_fla",
            "num_heads": values["raven_num_heads"],
            "num_kv_heads": values["raven_num_kv_heads"],
            "num_slots": values["raven_num_slots"],
            "topk": values["raven_topk"],
            "feature_map": values["raven_feature_map"],
            "decay_type": values["raven_decay_type"],
            "router_score": values["raven_router_score"],
            "router_type": values["raven_router_type"],
        },
        "gdn2_config": {
            "model_type": "gated_delta2_fla",
            "num_heads": values["gdn2_num_heads"],
            "num_v_heads": values["gdn2_num_v_heads"],
            "head_dim": values["gdn2_head_dim"],
        },
        "mamba3_config": {
            "model_type": "mamba3_fla",
            "head_dim": values["mamba3_head_dim"],
            "state_size": values["mamba3_state_size"],
            "is_mimo": values["mamba3_is_mimo"],
            "rescale_prenorm_residual": values["mamba3_rescale_prenorm_residual"],
        },
    }


def _pooling(values: Mapping[str, object]) -> ConfigMapping:
    return {
        "model_type": "gap_family",
        "pooling_type": values["pooling_type"],
        "input_size": values["hidden_size"],
        "output_size": values["pooling_output_size"],
        "num_heads": values["pooling_num_heads"],
        "gate_act": values["pooling_gate_act"],
        "hidden_ratio": values["pooling_hidden_ratio"],
        "output_act": values["pooling_output_act"],
        "use_output_norm": values["pooling_use_output_norm"],
        "dropout": values["pooling_dropout"],
    }


def _aasist(values: Mapping[str, object]) -> ConfigMapping:
    return {
        "model_type": "aasist",
        "input_size": values["aasist_input_size"],
        "filts": deepcopy(values["aasist_filts"]),
        "gat_dims": deepcopy(values["aasist_gat_dims"]),
        "pool_ratios": deepcopy(values["aasist_pool_ratios"]),
        "temperatures": deepcopy(values["aasist_temperatures"]),
        "graph_attention_dropout": values["aasist_graph_attention_dropout"],
        "graph_pool_dropout": values["aasist_graph_pool_dropout"],
        "path_dropout": values["aasist_path_dropout"],
        "output_dropout": values["aasist_output_dropout"],
    }


def _optimized_dataset_cache_path(value: object) -> object:
    """Preserve a legacy dataset path after moving configs into code_optimize."""

    if not isinstance(value, str) or Path(value).expanduser().is_absolute():
        return value
    return str(Path("..") / value)


def convert_legacy_config(values: Mapping[str, object]) -> ConfigMapping:
    """Convert one fully resolved flat training config to the optimized schema."""

    _validate_legacy_fields(values)
    pipeline = str(values["model_architecture"])
    model: ConfigMapping = {
        "model_architecture": pipeline,
        "frontend_config": _frontend(values),
    }
    if pipeline == "backbone_pooling":
        model.update(
            {
                "architecture_type": values["architecture_type"],
                "backbone_config": _backbone(values),
                "pooling_config": _pooling(values),
                "aasist_config": None,
                "classifier_config": _classifier(values, int(values["pooling_output_size"])),
            }
        )
    elif pipeline == "ssl_aasist":
        aasist = _aasist(values)
        model.update(
            {
                "architecture_type": "ssl_aasist",
                "backbone_config": None,
                "pooling_config": None,
                "aasist_config": aasist,
                "classifier_config": _classifier(
                    values,
                    5 * int(values["aasist_gat_dims"][1]),
                ),
            }
        )
    else:
        raise ValueError(f"Unsupported model_architecture={pipeline!r}.")

    return {
        "run": {
            "task_name": values["task_name"],
            "output_root": values["output_root"],
            "seed": values["seed"],
            "gpu_id": values["gpu_id"],
        },
        "data": {
            "dataset_key": values["dataset_key"],
            "dataset_cache_path": _optimized_dataset_cache_path(
                values["dataset_cache_path"]
            ),
            "train_split": values["train_split_name"],
            "validation_split": values["eval_split_name"],
            "input_column": values["input_values_column_name"],
            "label_column": values["label_column_name"],
        },
        "model": model,
        "loss": {
            "loss_type": values["loss_type"],
            "bonafide_weight": values["loss_bonafide_weight"],
            "spoof_weight": values["loss_spoof_weight"],
            "focal_gamma": values["focal_gamma"],
            "label_smoothing": values["ce_label_smoothing"],
        },
        "augmentation": {
            "algorithm": values["rawboost_algo"],
            "apply_to_validation": values["rawboost_apply_to_validation"],
            "sampling_rate": values["target_sampling_rate"],
            "num_bands": values["rawboost_num_bands"],
            "min_frequency": values["rawboost_min_frequency"],
            "max_frequency": values["rawboost_max_frequency"],
            "min_bandwidth": values["rawboost_min_bandwidth"],
            "max_bandwidth": values["rawboost_max_bandwidth"],
            "min_coefficients": values["rawboost_min_coefficients"],
            "max_coefficients": values["rawboost_max_coefficients"],
            "min_gain": values["rawboost_min_gain"],
            "max_gain": values["rawboost_max_gain"],
            "min_nonlinear_bias": values["rawboost_min_nonlinear_bias"],
            "max_nonlinear_bias": values["rawboost_max_nonlinear_bias"],
            "nonlinearity_order": values["rawboost_nonlinearity_order"],
            "impulse_percent": values["rawboost_impulse_percent"],
            "impulse_gain": values["rawboost_impulse_gain"],
            "min_snr": values["rawboost_min_snr"],
            "max_snr": values["rawboost_max_snr"],
        },
        "training": {
            "do_train": values["do_train"],
            "do_eval": values["do_eval"],
            "resume_from_checkpoint": None,
            "bf16": values["bf16"],
            "torch_compile": values["torch_compile"],
            "torch_compile_mode": values["torch_compile_mode"],
            "optimizer": values["optim"],
            "learning_rate": values["learning_rate"],
            "weight_decay": values["weight_decay"],
            "lr_scheduler_type": values["lr_scheduler_type"],
            "warmup_ratio": values["warmup_steps"],
            "adam_beta1": values["adam_beta1"],
            "adam_beta2": values["adam_beta2"],
            "num_train_epochs": values["num_train_epochs"],
            "per_device_train_batch_size": values["per_device_train_batch_size"],
            "per_device_eval_batch_size": values["per_device_eval_batch_size"],
            "gradient_accumulation_steps": 1,
            "dataloader_num_workers": values["dataloader_num_workers"],
            "remove_unused_columns": values["remove_unused_columns"],
            "eval_strategy": values["eval_strategy"],
            "save_strategy": values["save_strategy"],
            "logging_strategy": values["logging_strategy"],
            "logging_steps": values["logging_steps"],
            "save_total_limit": values["save_total_limit"],
            "load_best_model_at_end": values["load_best_model_at_end"],
            "metric_for_best_model": values["metric_for_best_model"],
            "greater_is_better": values["greater_is_better"],
            "early_stopping_patience": values["early_stopping_patience"],
            "early_stopping_threshold": values["early_stopping_threshold"],
            "report_to": deepcopy(values["report_to"]),
        },
    }


def _deep_diff(base: object, target: object) -> object:
    if not isinstance(base, Mapping) or not isinstance(target, Mapping):
        return deepcopy(target)
    if base.get("model_type") != target.get("model_type") and (
        "model_type" in base or "model_type" in target
    ):
        return deepcopy(dict(target))
    result: ConfigMapping = {}
    for key, target_value in target.items():
        if key not in base:
            result[key] = deepcopy(target_value)
            continue
        base_value = base[key]
        if isinstance(base_value, Mapping) and isinstance(target_value, Mapping):
            nested = _deep_diff(base_value, target_value)
            if nested:
                result[key] = nested
        elif base_value != target_value:
            result[key] = deepcopy(target_value)
    return result


def migrate_configs(
    source_root: Path,
    destination_root: Path,
    *,
    prune: bool = False,
) -> int:
    source_common = load_yaml_mapping(source_root / "train_common.yaml")
    source_paths = sorted((source_root / "experiments").rglob("*.yaml"))
    _validate_legacy_fields(
        source_common,
        source_name=source_root / "train_common.yaml",
    )
    source_overlays: list[tuple[Path, ConfigMapping]] = []
    for source_path in source_paths:
        source_overlay = load_yaml_mapping(source_path)
        _validate_legacy_fields(source_overlay, source_name=source_path)
        source_overlays.append((source_path, source_overlay))

    source_defaults = {
        "task_name": "baseline",
        "blocks_json": '{"0":{"attn":"raven","mlp":"mlp","num_layers":8}}',
        "hrm_h_blocks_json": None,
        "hrm_l_blocks_json": None,
        "architecture_type": "baseline",
        "looped_num_cycles": 2,
        "looped_num_gradient_cycles": 1,
        "hrm_h_cycles": 2,
        "hrm_l_cycles": 3,
        "hrm_num_gradient_steps": 2,
    }
    resolved_common = deep_merge(source_defaults, source_common)
    converted_common = convert_legacy_config(resolved_common)

    destination_root.mkdir(parents=True, exist_ok=True)
    experiments_destination = destination_root / "experiments"
    experiments_destination.mkdir(exist_ok=True)
    (destination_root / "common.yaml").write_text(
        yaml.safe_dump(converted_common, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    source_relatives = {path.relative_to(source_root / "experiments") for path in source_paths}
    if prune:
        for existing in experiments_destination.rglob("*.yaml"):
            if existing.relative_to(experiments_destination) not in source_relatives:
                existing.unlink()

    count = 0
    for source_path, source_overlay in source_overlays:
        resolved_source = deep_merge(resolved_common, source_overlay)
        converted = convert_legacy_config(resolved_source)
        overlay = _deep_diff(converted_common, converted)
        destination = experiments_destination / source_path.relative_to(source_root / "experiments")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Migrate legacy repository configs into the code_optimize/add_system schema.")
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPOSITORY_ROOT / "configs",
        help="Legacy repository config directory.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=CODE_OPTIMIZE_ROOT / "configs",
        help="Optimized config directory.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove optimized experiment YAML files that no longer exist in the source.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    count = migrate_configs(
        arguments.source.resolve(),
        arguments.destination.resolve(),
        prune=arguments.prune,
    )
    print(f"Migrated {count} experiment configurations.")


if __name__ == "__main__":
    main()
