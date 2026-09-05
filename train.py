from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from utilis.gpu_utils import (
    configure_torch_compile_stance_from_argv,
    configure_visible_gpu_from_argv,
)


CONFIGURED_GPU_ID = configure_visible_gpu_from_argv()
CONFIGURED_COMPILE_STANCE = configure_torch_compile_stance_from_argv()

import torch
from torch._dynamo import config as dynamo_config
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser, # 解析命令行参数到 dataclass
    TrainingArguments, # 控制训练、超参、评估、保存 checkpoint 等行为
    set_seed,
)
from transformers.trainer_utils import IntervalStrategy, get_last_checkpoint

from datasets import load_from_disk

from src.configuration import (
    ADDConfig,
    AASISTConfig,
    AttentionBackboneConfig,
    AttentionFLAConfig,
    GatedAttentionPoolingConfig,
    GatedDelta2FLAConfig,
    LinearClassifierConfig,
    LinearFrontendConfig,
    Mamba3FLAConfig,
    RavenFLAConfig,
    SincNetFrontendConfig,
    SSLFrontendConfig,
    SSL_FRONTEND_SPECS,
)
from src.modeling_add import ADDModel
from src.registration import register_add_for_auto
from utilis.config_utils import merge_yaml_configs, save_yaml_config
# Legacy experimental CUDA RawBoost implementation.  The import is retained as
# a reference for reproducing the old batch-side path, but it is intentionally
# disabled: training now uses the original CPU RawBoost transform below.
# from dev_functions.rawboost.cuda.rawboost_cuda_experimental import ExperimentalCudaRawBoostAlgo4
from utilis.rawboost_utils import (
    RawBoostArguments,
    RawBoostWaveformTransform,
    build_rawboost_dataset_transform,
)
from utilis.trainer_utils import (
    ADDTrainer,
    AudioClassificationCollator,
    LossArguments,
    build_loss_fn,
    compute_metrics,
    preprocess_logits_for_metrics,
)

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"tl\.make_block_ptr is deprecated.*",
    category=UserWarning,
    module=r"triton\.language\.core",
)

logger = logging.getLogger(__name__)


@dataclass
class RunArguments:
    task_name: str = field(
        default="need_specify_task_name",
        metadata={"help": "Task suffix. The final run directory is '<dataset_key>_<task_name>'."},
    )
    output_root: str = field(
        default="outputs",
        metadata={"help": "Root directory for default outputs. Used only when --output_dir is not set."},
    )
    gpu_id: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Physical GPU index exposed to this process through "
                "CUDA_VISIBLE_DEVICES. Configure it before torch is imported."
            )
        },
    )


@dataclass
class ModelArguments:
    model_architecture: str = field(
        default="backbone_pooling",
        metadata={"help": "Complete pipeline: backbone_pooling or ssl_aasist."},
    )
    frontend_type: str = field(
        default="linear",
        metadata={
            "help": (
                "linear, sincnet, wav2vec2-base, hubert-base-ls960, "
                "wavlm-base, or wavlm-base-plus."
            )
        },
    )
    architecture_type: str = field(
        default="baseline",
        metadata={"help": "Backbone architecture: baseline, hrm, heterogeneous_hrm, or looped."},
    )
    looped_num_cycles: int = field(
        default=2,
        metadata={"help": "N: number of shared-backbone forward cycles for looped architecture."},
    )
    looped_num_gradient_cycles: int = field(
        default=1,
        metadata={"help": "K: number of final looped cycles that participate in backpropagation."},
    )
    hrm_h_cycles: int = field(default=2, metadata={"help": "Number of HRM H cycles."})
    hrm_l_cycles: int = field(
        default=3,
        metadata={"help": "Number of L module calls inside each HRM H cycle."},
    )
    hrm_num_gradient_steps: int = field(
        default=2,
        metadata={"help": "K: number of final HRM module calls that participate in backpropagation."},
    )
    hidden_size: int = field(default=512)
    mlp_intermediate_size: int = field(default=1024) # 手动指定2倍
    num_labels: int = field(default=2)
    classifier_type: str = field(default="linear", metadata={"help": "linear or amsoftmax."})
    am_margin: float = field(default=0.3)
    am_scale: float = field(default=15.0)
    am_eps: float = field(default=1e-6)
    blocks_json: str = field(
        default=(
            '{"0":{"attn":"raven","mlp":"mlp","num_layers":8}}'
        ),
        metadata={"help": "JSON mapping/list used by baseline, looped, and homogeneous hrm."},
    )
    hrm_h_blocks_json: Optional[str] = field(
        default=None,
        metadata={"help": "H-module blocks required by heterogeneous_hrm."},
    )
    hrm_l_blocks_json: Optional[str] = field(
        default=None,
        metadata={"help": "L-module blocks required by heterogeneous_hrm."},
    )
    # frontend configs
    target_sampling_rate: int = field(default=16_000)
    frame_ms: float = field(default=25.0)
    window_ms: float = field(
        default=25.0,
        metadata={"help": "Only used for sincnet frontend. Sinc filter window length in milliseconds."},
    )
    stride_ms: float = field(
        default=20.0,
        metadata={"help": "Only used for sincnet frontend. Sinc filter stride in milliseconds."},
    )
    num_filters: int = field(default=80, metadata={"help": "Only used for sincnet frontend. Number of filters to learn."})
    cnn_kernels: list[int] = field(default_factory=lambda: [5, 3, 2, 2], metadata={"help": "Only used for sincnet frontend. List of kernel sizes for each CNN layer."})
    cnn_channels: list[int] = field(default_factory=lambda: [128, 256, 384, 512], metadata={"help": "Only used for sincnet frontend. List of output channels for each CNN layer. The last value will be overridden to match hidden_size."})
    sincnet_final_dropout: float = field(default=0.0)
    freeze_ssl_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Freeze pretrained SSL parameters. When selected layer indices "
                "are provided, only those encoder layers are re-enabled."
            )
        },
    )
    ssl_trainable_layer_indices: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": (
                "Zero-based SSL encoder layers to train while freeze_ssl_model=True. "
                "None keeps the existing full/frozen behavior."
            )
        },
    )
    # AASIST configs
    aasist_input_size: int = field(
        default=768,
        metadata={"help": "SSL feature size consumed by the AASIST input projection."},
    )
    aasist_filts: list[Any] = field(
        default_factory=lambda: [
            128,
            [1, 32],
            [32, 32],
            [32, 64],
            [64, 64],
        ]
    )
    aasist_gat_dims: list[int] = field(default_factory=lambda: [64, 32])
    aasist_pool_ratios: list[float] = field(
        default_factory=lambda: [0.5, 0.5, 0.5, 0.5]
    )
    aasist_temperatures: list[float] = field(
        default_factory=lambda: [2.0, 2.0, 100.0, 100.0]
    )
    aasist_graph_attention_dropout: float = field(default=0.2)
    aasist_graph_pool_dropout: float = field(default=0.3)
    aasist_path_dropout: float = field(default=0.2)
    aasist_output_dropout: float = field(default=0.5)
    # backbone configs
    raven_num_heads: int = field(default=8) # 4 or 8
    raven_num_kv_heads: Optional[int] = field(default=4, metadata={"help": "If not set, defaults to raven_num_heads."})
    raven_num_slots: Optional[int] = field(default=None)
    raven_topk: int = field(default=32)
    raven_feature_map: str = field(default="swish")
    raven_decay_type: str = field(default="Mamba2")
    raven_router_score: str = field(default="sigmoid")
    raven_router_type: str = field(default="lin")
    gdn2_num_heads: int = field(default=4)
    gdn2_num_v_heads: Optional[int] = field(default=8)
    gdn2_head_dim: int = field(default=64)
    mamba3_head_dim: int = field(default=64)
    mamba3_state_size: int = field(default=128)
    mamba3_is_mimo: bool = field(default=False)
    mamba3_rescale_prenorm_residual: bool = field(default=True)
    attention_num_heads: int = field(default=8)
    attention_num_kv_heads: Optional[int] = field(default=4, metadata={"help": "If not set, defaults to attention_num_heads."})
    attention_qk_norm: bool = field(default=True)
    attention_window_size: Optional[int] = field(default=None)
    attention_rope_theta: Optional[float] = field(default=10000.0)
    attention_max_position_embeddings: Optional[int] = field(default=None)
    residual_dropout: float = field(default=0.0)
    moe_aux_loss_coeff: float = field(default=0.01)
    moe_use_seq_aux_loss: bool = field(default=True)
    return_aux_loss: bool = field(default=True)
    # pooling configs
    pooling_type: str = field(
        default="mhgap",
        metadata={"help": "Pooling implementation name registered in src.pooling."},
    )
    pooling_output_size: int = field(default=128)
    pooling_num_heads: int = field(default=4)
    pooling_gate_act: str = field(default="sigmoid")
    pooling_hidden_ratio: float = field(default=2.0)
    pooling_output_act: str = field(default="none")
    pooling_use_output_norm: bool = field(default=False)
    pooling_dropout: float = field(default=0.0)


@dataclass
class DataArguments:
    dataset_key: str = field(
        default="asv19",
        metadata={"help": "Training dataset key. Currently only 'asv19' is supported."},
    )
    dataset_cache_path: str = field(
        default="../datasets/ASVspoof2019_16k_4s_fixed",
        metadata={"help": "Path passed to datasets.load_from_disk."},
    )
    train_split_name: str = field(default="train")
    eval_split_name: str = field(default="validation")
    input_values_column_name: str = field(default="input_values")
    label_column_name: str = field(default="labels")
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


def resolve_task_name(dataset_key: str, task_name: str) -> str:
    dataset_prefix = f"{dataset_key}_"
    if task_name.startswith(dataset_prefix):
        return task_name
    return f"{dataset_prefix}{task_name}"


@dataclass
class EarlyStoppingArguments:
    early_stopping_patience: int = field(
        default=7,
        metadata={"help": "Number of evaluation calls with no improvement before stopping. Set <= 0 to disable."},
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum metric improvement required to reset early stopping patience."},
    )


def parse_blocks(
    blocks_json: str,
    argument_name: str = "blocks_json",
) -> dict[Any, Any] | list[Any]:
    try:
        blocks = json.loads(blocks_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{argument_name} is not valid JSON: {exc}") from exc
    if not isinstance(blocks, (dict, list)):
        raise TypeError(f"{argument_name} must decode to a dict or list.")
    return blocks


def build_add_config(model_args: ModelArguments) -> ADDConfig:
    if model_args.model_architecture not in {"backbone_pooling", "ssl_aasist"}:
        raise ValueError(
            "model_architecture must be one of backbone_pooling or ssl_aasist, "
            f"got {model_args.model_architecture!r}."
        )
    if (
        model_args.model_architecture == "ssl_aasist"
        and model_args.frontend_type not in SSL_FRONTEND_SPECS
    ):
        supported = ", ".join(SSL_FRONTEND_SPECS)
        raise ValueError(
            "model_architecture='ssl_aasist' requires an SSL frontend; "
            f"expected one of {supported}."
        )

    if model_args.frontend_type == "linear":
        frontend_config = LinearFrontendConfig(
            target_sampling_rate=model_args.target_sampling_rate,
            frame_ms=model_args.frame_ms,
            output_size=model_args.hidden_size,
        )
    elif model_args.frontend_type == "sincnet":
        cnn_channels = list(model_args.cnn_channels)
        if not cnn_channels:
            raise ValueError("cnn_channels must not be empty.")
        cnn_channels[-1] = model_args.hidden_size
        frontend_config = SincNetFrontendConfig(
            target_sampling_rate=model_args.target_sampling_rate,
            window_ms=model_args.window_ms,
            stride_ms=model_args.stride_ms,
            num_filters=model_args.num_filters,
            cnn_kernels=model_args.cnn_kernels,
            cnn_channels=cnn_channels,
            final_dropout=model_args.sincnet_final_dropout,
        )
    elif model_args.frontend_type in SSL_FRONTEND_SPECS:
        ssl_output_size = (
            model_args.aasist_input_size
            if model_args.model_architecture == "ssl_aasist"
            else model_args.hidden_size
        )
        frontend_config = SSLFrontendConfig(
            ssl_type=model_args.frontend_type,
            target_sampling_rate=model_args.target_sampling_rate,
            output_size=ssl_output_size,
            freeze_ssl_model=model_args.freeze_ssl_model,
            ssl_trainable_layer_indices=model_args.ssl_trainable_layer_indices,
        )
    else:
        supported = ", ".join(("linear", "sincnet", *SSL_FRONTEND_SPECS))
        raise ValueError(
            f"Unsupported frontend_type={model_args.frontend_type!r}; "
            f"expected one of {supported}."
        )

    if model_args.model_architecture == "ssl_aasist":
        aasist_config = AASISTConfig(
            input_size=model_args.aasist_input_size,
            filts=model_args.aasist_filts,
            gat_dims=model_args.aasist_gat_dims,
            pool_ratios=model_args.aasist_pool_ratios,
            temperatures=model_args.aasist_temperatures,
            graph_attention_dropout=model_args.aasist_graph_attention_dropout,
            graph_pool_dropout=model_args.aasist_graph_pool_dropout,
            path_dropout=model_args.aasist_path_dropout,
            output_dropout=model_args.aasist_output_dropout,
        )
        classifier_config = LinearClassifierConfig(
            input_size=aasist_config.output_size,
            num_labels=model_args.num_labels,
            classifier_type=model_args.classifier_type,
            am_margin=model_args.am_margin,
            am_scale=model_args.am_scale,
            am_eps=model_args.am_eps,
        )
        config = ADDConfig(
            model_architecture="ssl_aasist",
            frontend_config=frontend_config,
            backbone_config=None,
            pooling_config=None,
            aasist_config=aasist_config,
            classifier_config=classifier_config,
        )
        config.has_no_defaults_at_init = True
        return config

    heterogeneous_hrm = model_args.architecture_type == "heterogeneous_hrm"
    blocks = None if heterogeneous_hrm else parse_blocks(model_args.blocks_json)
    hrm_h_blocks = (
        parse_blocks(model_args.hrm_h_blocks_json, "hrm_h_blocks_json")
        if model_args.hrm_h_blocks_json is not None
        else None
    )
    hrm_l_blocks = (
        parse_blocks(model_args.hrm_l_blocks_json, "hrm_l_blocks_json")
        if model_args.hrm_l_blocks_json is not None
        else None
    )

    backbone_config = AttentionBackboneConfig(
        hidden_size=model_args.hidden_size,
        architecture_type=model_args.architecture_type,
        looped_num_cycles=model_args.looped_num_cycles,
        looped_num_gradient_cycles=model_args.looped_num_gradient_cycles,
        hrm_h_cycles=model_args.hrm_h_cycles,
        hrm_l_cycles=model_args.hrm_l_cycles,
        hrm_num_gradient_steps=model_args.hrm_num_gradient_steps,
        mlp_intermediate_size=model_args.mlp_intermediate_size,
        blocks=blocks,
        hrm_h_blocks=hrm_h_blocks,
        hrm_l_blocks=hrm_l_blocks,
        attention_config=AttentionFLAConfig(
            num_heads=model_args.attention_num_heads,
            num_kv_heads=model_args.attention_num_kv_heads,
            qk_norm=model_args.attention_qk_norm,
            window_size=model_args.attention_window_size,
            rope_theta=model_args.attention_rope_theta,
            max_position_embeddings=model_args.attention_max_position_embeddings,
        ),
        raven_config=RavenFLAConfig(
            num_heads=model_args.raven_num_heads,
            num_kv_heads=model_args.raven_num_kv_heads,
            num_slots=model_args.raven_num_slots,
            topk=model_args.raven_topk,
            feature_map=model_args.raven_feature_map,
            decay_type=model_args.raven_decay_type,
            router_score=model_args.raven_router_score,
            router_type=model_args.raven_router_type,
        ),
        gdn2_config=GatedDelta2FLAConfig(
            num_heads=model_args.gdn2_num_heads,
            num_v_heads=model_args.gdn2_num_v_heads,
            head_dim=model_args.gdn2_head_dim,
        ),
        mamba3_config=Mamba3FLAConfig(
            head_dim=model_args.mamba3_head_dim,
            state_size=model_args.mamba3_state_size,
            is_mimo=model_args.mamba3_is_mimo,
            rescale_prenorm_residual=model_args.mamba3_rescale_prenorm_residual,
        ),
        residual_dropout=model_args.residual_dropout,
        moe_aux_loss_coeff=model_args.moe_aux_loss_coeff,
        moe_use_seq_aux_loss=model_args.moe_use_seq_aux_loss,
        return_aux_loss=model_args.return_aux_loss,
    )
    pooling_config = GatedAttentionPoolingConfig(
        pooling_type=model_args.pooling_type,
        input_size=model_args.hidden_size,
        output_size=model_args.pooling_output_size,
        num_heads=model_args.pooling_num_heads,
        gate_act=model_args.pooling_gate_act,
        hidden_ratio=model_args.pooling_hidden_ratio,
        output_act=model_args.pooling_output_act,
        use_output_norm=model_args.pooling_use_output_norm,
        dropout=model_args.pooling_dropout,
    )
    classifier_config = LinearClassifierConfig(
        input_size=model_args.pooling_output_size,
        num_labels=model_args.num_labels,
        classifier_type=model_args.classifier_type,
        am_margin=model_args.am_margin,
        am_scale=model_args.am_scale,
        am_eps=model_args.am_eps,
    )
    config = ADDConfig(
        model_architecture="backbone_pooling",
        architecture_type=model_args.architecture_type,
        frontend_config=frontend_config,
        backbone_config=backbone_config,
        pooling_config=pooling_config,
        classifier_config=classifier_config,
    )
    config.has_no_defaults_at_init = True
    return config


def load_audio_datasets(
    data_args: DataArguments,
    rawboost_args: RawBoostArguments,
    sampling_rate: int,
    training_args: TrainingArguments,
):
    if training_args.do_train and data_args.dataset_key != "asv19":
        raise ValueError("Training currently only supports dataset_key='asv19'.")

    ds_cache = load_from_disk(data_args.dataset_cache_path)
    train_dataset = ds_cache[data_args.train_split_name] if training_args.do_train else None
    eval_dataset = ds_cache[data_args.eval_split_name] if training_args.do_eval else None

    if train_dataset is not None and data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(data_args.max_train_samples, len(train_dataset))))
    if eval_dataset is not None and data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(data_args.max_eval_samples, len(eval_dataset))))

    if rawboost_args.enabled:
        # The cached samples are already trimmed/repeated to four seconds. RawBoost
        # therefore augments the final model input rather than the original waveform.
        waveform_transform = RawBoostWaveformTransform(rawboost_args, sampling_rate)
        dataset_transform = build_rawboost_dataset_transform(
            waveform_transform,
            data_args.input_values_column_name,
        )
        if train_dataset is not None:
            train_dataset = train_dataset.with_transform(dataset_transform)
        if eval_dataset is not None and rawboost_args.rawboost_apply_to_validation:
            eval_dataset = eval_dataset.with_transform(dataset_transform)
        logger.info(
            "RawBoost enabled: algo=%s, train=%s, validation=%s, sampling_rate=%s",
            rawboost_args.rawboost_algo,
            train_dataset is not None,
            eval_dataset is not None and rawboost_args.rawboost_apply_to_validation,
            sampling_rate,
        )
    else:
        logger.info("RawBoost disabled (rawboost_algo=0).")

    # Legacy CUDA algo4 path (intentionally disabled).  This used to leave the
    # dataset untouched and augment each training batch inside ADDTrainer.  All
    # three matching call sites are kept below as comments so the experiment can
    # be reconstructed without making CUDA RawBoost part of the normal pipeline.
    # if rawboost_args.rawboost_algo == 4:
    #     logger.info(
    #         "RawBoost algo4 deferred to experimental CUDA batch augmentation: "
    #         "train=%s, validation=%s, sampling_rate=%s",
    #         train_dataset is not None,
    #         eval_dataset is not None and rawboost_args.rawboost_apply_to_validation,
    #         sampling_rate,
    #     )

    return train_dataset, eval_dataset


def _format_trainable_parameters(module) -> str:
    if not hasattr(module, "parameter_name_summary"):
        return "  parameter_name_summary: not available"

    lines = []
    for item in module.parameter_name_summary():
        if not item.get("requires_grad", False):
            continue
        name = item["name"]
        shape = item["shape"]
        dtype = item["dtype"]
        numel = module.get_parameter(name).numel()
        lines.append(f"  {name}: shape={shape}, dtype={dtype}, numel={numel}")
    return "\n".join(lines) if lines else "  no trainable parameters"


def write_model_overview(model: ADDModel, output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    overview_path = output_path / "model_overview.txt"

    sections = [("frontend", model.frontend)]
    if model.backbone is not None:
        sections.append(("backbone", model.backbone))
    if model.pooling is not None:
        sections.append(("pooling", model.pooling))
    if model.aasist is not None:
        sections.append(("aasist", model.aasist))
    sections.append(("classifier", model.classifier))

    lines = ["ADDModel Overview", ""]
    total_trainable = 0
    for title, module in sections:
        trainable_count = module.num_trainable_parameters()
        total_trainable += trainable_count
        lines.extend(
            [
                "=" * 80,
                title,
                "=" * 80,
                "",
                "Architecture",
                "-" * 80,
                module.architecture_summary(),
                "",
                f"Trainable parameters: {trainable_count}",
                "-" * 80,
                _format_trainable_parameters(module),
                "",
            ]
        )

    lines.extend(
        [
            "=" * 80,
            f"Total trainable parameters: {total_trainable}",
            "=" * 80,
            "",
        ]
    )
    overview_path.write_text("\n".join(lines), encoding="utf-8")
    return overview_path


def parse_train_arguments(argv: Optional[list[str]] = None):
    parser = HfArgumentParser(
        (
            RunArguments,
            ModelArguments,
            LossArguments,
            RawBoostArguments,
            DataArguments,
            EarlyStoppingArguments,
            TrainingArguments,
        )
    )
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_parser.add_argument("--experiment-config")
    config_parser.add_argument("--gpu_id", type=int)
    config_args, remaining_args = config_parser.parse_known_args(argv)

    config_paths = [
        path
        for path in (config_args.config, config_args.experiment_config)
        if path is not None
    ]
    if config_paths:
        if remaining_args:
            raise ValueError(
                "YAML mode does not accept additional CLI arguments. "
                f"Unexpected arguments: {remaining_args}"
            )
        resolved_config = merge_yaml_configs(*config_paths)
        if config_args.gpu_id is not None:
            resolved_config["gpu_id"] = config_args.gpu_id
        if resolved_config.get("torch_compile") is False:
            # Transformers enables torch_compile again whenever either of
            # these optional settings remains non-null.
            resolved_config["torch_compile_backend"] = None
            resolved_config["torch_compile_mode"] = None
        parsed = parser.parse_dict(resolved_config, allow_extra_keys=False)
        return (*parsed, resolved_config)

    parsed = parser.parse_args_into_dataclasses(args=argv)
    return (*parsed, None)


def main() -> None:
    (
        run_args,
        model_args,
        loss_args,
        rawboost_args,
        data_args,
        early_stopping_args,
        training_args,
        resolved_config,
    ) = parse_train_arguments()

    if training_args.torch_compile:
        compile_mode = training_args.torch_compile_mode or "default"
        supported_compile_modes = {"default", "max-autotune-no-cudagraphs"}
        if compile_mode not in supported_compile_modes:
            supported = ", ".join(sorted(supported_compile_modes))
            raise ValueError(
                f"torch_compile_mode must be one of {supported}; got {compile_mode!r}."
            )
        training_args.torch_compile_mode = compile_mode
        compile_stance = "default"
    else:
        compile_stance = "force_eager"
    if compile_stance != CONFIGURED_COMPILE_STANCE:
        raise RuntimeError(
            "torch_compile changed after startup configuration: "
            f"startup_stance={CONFIGURED_COMPILE_STANCE!r}, "
            f"parsed_stance={compile_stance!r}."
        )

    capture_ssl_specaugment_shapes = (
        training_args.torch_compile
        and model_args.frontend_type in SSL_FRONTEND_SPECS
        and not model_args.freeze_ssl_model
    )
    dynamo_config.capture_dynamic_output_shape_ops = capture_ssl_specaugment_shapes
    logger.info(
        "torch.compile: enabled=%s, stance=%s, backend=%s, mode=%s, "
        "capture_ssl_specaugment_shapes=%s",
        training_args.torch_compile,
        compile_stance,
        training_args.torch_compile_backend or "inductor",
        training_args.torch_compile_mode or "default",
        capture_ssl_specaugment_shapes,
    )

    resolved_task_name = resolve_task_name(data_args.dataset_key, run_args.task_name)
    if training_args.output_dir in {None, "trainer_output"}:
        training_args.output_dir = str(Path(run_args.output_root) / resolved_task_name)
    if resolved_config is not None:
        resolved_config = dict(resolved_config)
        resolved_config["output_dir"] = training_args.output_dir
        save_yaml_config(
            resolved_config,
            Path(training_args.output_dir) / "resolved_config.yaml",
        )
    logger.info("Current task: %s", resolved_task_name)
    logger.info("Task name argument: %s", run_args.task_name)
    logger.info("Dataset key: %s", data_args.dataset_key)
    logger.info("Output directory: %s", training_args.output_dir)
    if run_args.gpu_id != CONFIGURED_GPU_ID:
        raise RuntimeError(
            "gpu_id changed after CUDA startup configuration: "
            f"startup={CONFIGURED_GPU_ID!r}, parsed={run_args.gpu_id!r}."
        )
    if not torch.cuda.is_available():
        if run_args.gpu_id is not None:
            raise RuntimeError(
                f"Configured physical GPU {run_args.gpu_id}, but CUDA is unavailable. "
                "Check nvidia-smi and the configured gpu_id."
            )
        raise RuntimeError("ADDModel training expects CUDA because the FLA kernels are GPU-oriented.")
    logger.info(
        "GPU assignment: configured_gpu_id=%s, CUDA_VISIBLE_DEVICES=%s, "
        "process_device=cuda:0, device_name=%s",
        run_args.gpu_id,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"),
        torch.cuda.get_device_name(0),
    )
    # if training_args.remove_unused_columns:
    #     logger.warning("Setting remove_unused_columns=False because ADDTrainer consumes labels in compute_loss.")
    #     training_args.remove_unused_columns = False

    set_seed(training_args.seed)
    register_add_for_auto()

    config = build_add_config(model_args)
    model = ADDModel(config)
    if model.backbone is not None:
        logger.info(
            "Architecture: pipeline=%s, type=%s, recurrent_cycles=%s, "
            "gradient_cycles=%s, post_norm_enabled=%s",
            config.model_architecture,
            config.architecture_type,
            model.backbone.num_recurrent_cycles,
            model.backbone.num_gradient_cycles,
            model.backbone.use_post_norm,
        )
    else:
        logger.info(
            "Architecture: pipeline=%s, SSL output=%s, AASIST output=%s",
            config.model_architecture,
            config.frontend_config.frontend_output_dim,
            config.aasist_config.output_size,
        )
    loss_fn = build_loss_fn(loss_args, model_args.num_labels)
    logger.info(
        "Loss: type=%s, class_weights=%s, focal_gamma=%s, ce_label_smoothing=%s",
        loss_args.loss_type,
        loss_args.class_weights,
        loss_args.focal_gamma,
        loss_args.ce_label_smoothing,
    )
    train_dataset, eval_dataset = load_audio_datasets(
        data_args,
        rawboost_args,
        model_args.target_sampling_rate,
        training_args,
    )
    data_collator = AudioClassificationCollator(
        input_values_column_name=data_args.input_values_column_name,
        label_column_name=data_args.label_column_name,
    )

    # Legacy CUDA RawBoost construction (disabled; see load_audio_datasets).
    # cuda_rawboost = (
    #     ExperimentalCudaRawBoostAlgo4(
    #         config=rawboost_args,
    #         sampling_rate=model_args.target_sampling_rate,
    #     )
    #     if rawboost_args.rawboost_algo == 4
    #     else None
    # )

    overview_path = write_model_overview(model, training_args.output_dir)
    # logger.info("Model overview written to %s", overview_path)
    # logger.info("Backbone architecture:\n%s", model.backbone.architecture_summary())
    callbacks = []
    if training_args.do_train and early_stopping_args.early_stopping_patience > 0:
        if not training_args.do_eval:
            raise ValueError("Early stopping requires do_eval=True and an eval_dataset.")
        if training_args.eval_strategy == IntervalStrategy.NO:
            raise ValueError("Early stopping requires TrainingArguments.eval_strategy to be 'steps' or 'epoch'.")
        if training_args.metric_for_best_model is None:
            training_args.metric_for_best_model = "eval_loss"
        if training_args.greater_is_better is None:
            training_args.greater_is_better = False
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_args.early_stopping_patience,
                early_stopping_threshold=early_stopping_args.early_stopping_threshold,
            )
        )

    trainer = ADDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
        loss_fn=loss_fn,
        # Legacy CUDA RawBoost Trainer arguments (disabled; CPU RawBoost is
        # attached to Dataset.with_transform instead).
        # cuda_rawboost=cuda_rawboost,
        # rawboost_apply_to_validation=rawboost_args.rawboost_apply_to_validation,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics) # 控制台输出
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        logger.info("Best checkpoint: %s", trainer.state.best_model_checkpoint)
        logger.info("Last checkpoint: %s", last_checkpoint)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
