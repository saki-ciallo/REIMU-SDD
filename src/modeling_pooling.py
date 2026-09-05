from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration import GatedAttentionPoolingConfig
from .pooling import (
    GatedAttentionPooling,
    GatedAttentionPoolingV2,
    GlobalAvgMaxPooling,
    POOLING_REGISTRY,
    MultiHeadGatedAttentionPooling,
    build_pooling,
)


@dataclass
class GatedAttentionPoolingOutput(ModelOutput):
    pooled_output: Optional[torch.Tensor] = None


class GatedAttentionPoolingSummaryMixin:
    def parameter_name_summary(self) -> List[Dict[str, object]]:
        return [
            {
                "name": name,
                "shape": tuple(parameter.shape),
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in self.named_parameters()
        ]

    def architecture_summary(self) -> str:
        pooling = self.pooling
        lines = [
            "PoolingModel",
            f"  pooling_type: {self.config.pooling_type}",
            (
                "  pooling: "
                f"{pooling.__class__.__name__}(input_size={self.config.input_size}, "
                f"output_size={self.config.output_size})"
            ),
        ]
        if isinstance(pooling, MultiHeadGatedAttentionPooling):
            lines.extend(
                [
                    f"    num_heads: {pooling.num_heads}",
                    f"    head_dim: {pooling.head_dim}",
                    f"    value_gate_proj: Linear({pooling.input_size}->{2 * pooling.output_size})",
                    f"    gate_act: {pooling.gate_act}",
                    f"    score_temperature: {pooling.score_temperature}",
                    f"    score_weight: Parameter({pooling.num_heads}, {pooling.head_dim})",
                    f"    output_act: {pooling.output_act.__class__.__name__}",
                    (
                        f"    output_norm: {pooling.output_norm.__class__.__name__}"
                        f"(enabled={pooling.use_output_norm})"
                    ),
                    (
                        f"    dropout: {pooling.dropout.__class__.__name__}"
                        f"(p={pooling.dropout.p if isinstance(pooling.dropout, nn.Dropout) else 0.0})"
                    ),
                ]
            )
        elif isinstance(pooling, GlobalAvgMaxPooling):
            lines.extend(
                [
                    "    aggregation: concat(mean, max)",
                    f"    fused_proj: Linear({2 * pooling.input_size}->{pooling.output_size})",
                    f"    output_act: {pooling.output_act.__class__.__name__}",
                    (
                        f"    output_norm: {pooling.output_norm.__class__.__name__}"
                        f"(enabled={pooling.use_output_norm})"
                    ),
                    (
                        f"    dropout: {pooling.dropout.__class__.__name__}"
                        f"(p={pooling.dropout.p if isinstance(pooling.dropout, nn.Dropout) else 0.0})"
                    ),
                ]
            )
        elif isinstance(pooling, GatedAttentionPooling):
            lines.extend(
                [
                    f"    hidden_ratio: {pooling.hidden_ratio}",
                    f"    hidden_dim: {pooling.hidden_dim}",
                    f"    gate_value_proj: Linear({pooling.input_size}->{2 * pooling.hidden_dim})",
                    f"    to_score: Linear({pooling.hidden_dim}->1)",
                    f"    score_temperature: {pooling.score_temperature}",
                    f"    output_proj: {pooling.output_proj.__class__.__name__}",
                    f"    output_act: {pooling.output_act.__class__.__name__}",
                    (
                        f"    output_norm: {pooling.output_norm.__class__.__name__}"
                        f"(enabled={pooling.use_output_norm})"
                    ),
                    (
                        f"    dropout: {pooling.dropout.__class__.__name__}"
                        f"(p={pooling.dropout.p if isinstance(pooling.dropout, nn.Dropout) else 0.0})"
                    ),
                ]
            )
        elif isinstance(pooling, GatedAttentionPoolingV2):
            lines.extend(
                [
                    f"    hidden_ratio: {pooling.hidden_ratio}",
                    f"    hidden_dim: {pooling.hidden_dim}",
                    f"    proj: Linear({pooling.input_size}->{3 * pooling.hidden_dim})",
                    f"    to_score: Linear({pooling.hidden_dim}->1)",
                    f"    score_temperature: {pooling.score_temperature}",
                    f"    output_proj: Linear({pooling.hidden_dim}->{pooling.output_size})",
                    f"    output_act: {pooling.output_act.__class__.__name__}",
                    (
                        f"    output_norm: {pooling.output_norm.__class__.__name__}"
                        f"(enabled={pooling.use_output_norm})"
                    ),
                    (
                        f"    dropout: {pooling.dropout.__class__.__name__}"
                        f"(p={pooling.dropout.p if isinstance(pooling.dropout, nn.Dropout) else 0.0})"
                    ),
                ]
            )
        return "\n".join(lines)

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class GatedAttentionPoolingModel(GatedAttentionPoolingSummaryMixin, PreTrainedModel):
    config_class = GatedAttentionPoolingConfig
    base_model_prefix = "pooling"
    _no_split_modules = [pooling_cls.__name__ for pooling_cls in POOLING_REGISTRY.values()]

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__(config)
        self.pooling = build_pooling(config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, MultiHeadGatedAttentionPooling):
            nn.init.normal_(module.score_weight, mean=0.0, std=std)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm) or module.__class__.__name__ == "RMSNorm":
            nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> tuple | GatedAttentionPoolingOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        hidden_states = input_ids
        pooled_output = self.pooling(hidden_states=hidden_states)

        if not return_dict:
            return (pooled_output,)

        return GatedAttentionPoolingOutput(
            pooled_output=pooled_output,
        )


__all__ = [
    "GatedAttentionPoolingModel",
    "GatedAttentionPoolingOutput",
    "GatedAttentionPoolingSummaryMixin",
]
