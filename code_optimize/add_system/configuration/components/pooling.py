from __future__ import annotations

from enum import StrEnum
from typing import Any

from transformers import PretrainedConfig

from ..validation import boolean, one_of, positive_float, positive_int, probability


class PoolingType(StrEnum):
    MHGAP = "mhgap"
    GAMP = "gamp"
    GAPV1 = "gapv1"
    GAPV2 = "gapv2"


class GatedAttentionPoolingConfig(PretrainedConfig):
    """Configuration shared by the pooling-family implementations."""

    model_type = "gap_family"
    legacy_model_types = ("mhgap", "gated_attention_pooling")

    def __init__(
        self,
        pooling_type: str = "mhgap",
        input_size: int = 192,
        output_size: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
        gate_act: str = "silu",
        score_temperature: float = 1.0,
        hidden_ratio: float = 2.0,
        output_act: str = "none",
        use_output_norm: bool = False,
        layer_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pooling_type = PoolingType(
            one_of(
                pooling_type,
                {member.value for member in PoolingType},
                field_name="pooling_type",
            )
        ).value
        self.input_size = positive_int(input_size, field_name="input_size")
        self.output_size = positive_int(output_size, field_name="output_size")
        self.num_heads = positive_int(num_heads, field_name="num_heads")
        if self.pooling_type == PoolingType.MHGAP and self.output_size % self.num_heads:
            raise ValueError("MHGAP output_size must be divisible by num_heads.")
        self.dropout = probability(dropout, field_name="dropout")
        self.bias = boolean(bias, field_name="bias")
        self.gate_act = one_of(
            gate_act,
            {"sigmoid", "silu"},
            field_name="gate_act",
        )
        self.score_temperature = positive_float(
            score_temperature,
            field_name="score_temperature",
        )
        self.hidden_ratio = positive_float(hidden_ratio, field_name="hidden_ratio")
        self.output_act = one_of(
            output_act,
            {"none", "silu"},
            field_name="output_act",
        )
        self.use_output_norm = boolean(
            use_output_norm,
            field_name="use_output_norm",
        )
        self.layer_norm_eps = positive_float(
            layer_norm_eps,
            field_name="layer_norm_eps",
        )
        self.initializer_range = positive_float(
            initializer_range,
            field_name="initializer_range",
        )

    @property
    def resolved_num_heads(self) -> int:
        return self.num_heads

    @property
    def resolved_head_dim(self) -> int:
        if self.output_size % self.num_heads:
            raise ValueError("output_size must be divisible by num_heads.")
        return self.output_size // self.num_heads
