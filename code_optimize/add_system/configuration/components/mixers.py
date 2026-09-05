from __future__ import annotations

from typing import Any, ClassVar

from transformers import PretrainedConfig

from ..validation import boolean, one_of, optional_positive_int, positive_float, positive_int


class RavenFLAConfig(PretrainedConfig):
    model_type = "raven_fla"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        num_heads: int = 4,
        num_kv_heads: int | None = 2,
        num_slots: int | None = None,
        topk: int = 32,
        feature_map: str = "swish",
        decay_type: str = "Mamba2",
        router_score: str = "sigmoid",
        router_type: str = "lin",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = positive_int(num_heads, field_name="num_heads")
        self.num_kv_heads = optional_positive_int(
            num_kv_heads,
            field_name="num_kv_heads",
        )
        if self.num_kv_heads is not None and self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        self.num_slots = optional_positive_int(num_slots, field_name="num_slots")
        self.topk = positive_int(topk, field_name="topk")
        if self.num_slots is not None and self.topk > self.num_slots:
            raise ValueError("topk must not exceed num_slots.")
        self.feature_map = feature_map
        self.decay_type = decay_type
        self.router_score = one_of(
            router_score,
            {"sigmoid", "softmax"},
            field_name="router_score",
        )
        self.router_type = one_of(router_type, {"lin", "mlp"}, field_name="router_type")


class GatedDelta2FLAConfig(PretrainedConfig):
    model_type = "gated_delta2_fla"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        num_heads: int = 4,
        num_v_heads: int | None = 8,
        head_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = positive_int(num_heads, field_name="num_heads")
        self.num_v_heads = optional_positive_int(num_v_heads, field_name="num_v_heads")
        if self.num_v_heads is not None and self.num_v_heads % self.num_heads:
            raise ValueError("num_v_heads must be divisible by num_heads.")
        self.head_dim = positive_int(head_dim, field_name="head_dim")


class Mamba3FLAConfig(PretrainedConfig):
    model_type = "mamba3_fla"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        head_dim: int = 64,
        state_size: int = 128,
        is_mimo: bool = False,
        rescale_prenorm_residual: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.head_dim = positive_int(head_dim, field_name="head_dim")
        self.state_size = positive_int(state_size, field_name="state_size")
        self.is_mimo = boolean(is_mimo, field_name="is_mimo")
        self.rescale_prenorm_residual = boolean(
            rescale_prenorm_residual,
            field_name="rescale_prenorm_residual",
        )


class AttentionFLAConfig(PretrainedConfig):
    model_type = "attention_fla"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        num_heads: int = 4,
        num_kv_heads: int | None = 2,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10_000.0,
        max_position_embeddings: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = positive_int(num_heads, field_name="num_heads")
        self.num_kv_heads = optional_positive_int(
            num_kv_heads,
            field_name="num_kv_heads",
        )
        if self.num_kv_heads is not None and self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        self.qk_norm = boolean(qk_norm, field_name="qk_norm")
        self.window_size = optional_positive_int(window_size, field_name="window_size")
        self.rope_theta = (
            positive_float(rope_theta, field_name="rope_theta") if rope_theta is not None else None
        )
        self.max_position_embeddings = optional_positive_int(
            max_position_embeddings,
            field_name="max_position_embeddings",
        )
