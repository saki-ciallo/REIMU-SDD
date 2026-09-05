from __future__ import annotations

from typing import Any, Optional, Sequence

from transformers import PretrainedConfig


def _config_from_dict(config: Any, config_classes: tuple[type[PretrainedConfig], ...]) -> Any:
    if not isinstance(config, dict):
        return config
    model_type = config.get("model_type")
    for config_class in config_classes:
        compatible_model_types = {
            config_class.model_type,
            *getattr(config_class, "legacy_model_types", ()),
        }
        if model_type in compatible_model_types:
            return config_class(**config)
    expected = ", ".join(
        model_type
        for config_class in config_classes
        for model_type in (
            config_class.model_type,
            *getattr(config_class, "legacy_model_types", ()),
        )
    )
    raise TypeError(f"Unsupported nested config model_type={model_type!r}; expected one of {expected}.")

class SincNetFrontendConfig(PretrainedConfig):
    """Configuration for the SincNet audio frontend."""

    model_type = "sincnet_frontend"

    def __init__(
        self,
        target_sampling_rate: int = 16_000,
        window_ms: float = 25.0,
        stride_ms: float = 20.0,
        num_filters: int = 80,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
        cnn_kernels: Optional[Sequence[int]] = None,
        cnn_channels: Optional[Sequence[int]] = None,
        cnn_bias: bool = False,
        apply_final_cnn_activation: bool = False,
        use_final_sinc_residual: bool = False, # 一定关闭，不要修改
        final_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Frontend: waveform -> SincConv -> CNN features.
        self.target_sampling_rate = target_sampling_rate
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.num_filters = num_filters
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.cnn_kernels = list(cnn_kernels) if cnn_kernels is not None else [5, 3]
        self.cnn_channels = list(cnn_channels) if cnn_channels is not None else [128, 192]
        self.cnn_bias = cnn_bias
        self.apply_final_cnn_activation = apply_final_cnn_activation
        self.use_final_sinc_residual = use_final_sinc_residual
        if not 0.0 <= final_dropout < 1.0:
            raise ValueError("final_dropout must satisfy 0.0 <= final_dropout < 1.0.")
        self.final_dropout = final_dropout
        self.rms_norm_eps = rms_norm_eps

    @property
    def window_size(self) -> int:
        return int(round(self.target_sampling_rate * self.window_ms / 1000.0))

    @property
    def stride_size(self) -> int:
        return int(round(self.target_sampling_rate * self.stride_ms / 1000.0))

    @property
    def overlap_size(self) -> int:
        return self.window_size - self.stride_size

    @property
    def frontend_output_dim(self) -> int:
        return self.cnn_channels[-1]


class LinearFrontendConfig(PretrainedConfig):
    """
    Configuration for non-overlapping frame slicing + linear projection.
    Gemma 4 12B like preprocessing frontend.
    """

    model_type = "linear_frontend"

    def __init__(
        self,
        target_sampling_rate: int = 16_000,
        frame_ms: float = 25.0,
        output_size: int = 512,
        bias: bool = False,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_sampling_rate = target_sampling_rate
        self.frame_ms = frame_ms
        self.output_size = output_size
        self.bias = bias
        self.initializer_range = initializer_range

    @property
    def frame_size(self) -> int:
        return int(round(self.target_sampling_rate * self.frame_ms / 1000.0))

    @property
    def frontend_output_dim(self) -> int:
        return self.output_size


SSL_FRONTEND_SPECS = {
    "wav2vec2-base": {
        "pretrained_model_name_or_path": "facebook/wav2vec2-base",
        "hidden_size": 768,
        "sampling_rate": 16_000,
        "normalize_input": True,
    },
    "hubert-base-ls960": {
        "pretrained_model_name_or_path": "facebook/hubert-base-ls960",
        "hidden_size": 768,
        "sampling_rate": 16_000,
        "normalize_input": True,
    },
    "wavlm-base": {
        "pretrained_model_name_or_path": "microsoft/wavlm-base",
        "hidden_size": 768,
        "sampling_rate": 16_000,
        "normalize_input": False,
    },
    "wavlm-base-plus": {
        "pretrained_model_name_or_path": "microsoft/wavlm-base-plus",
        "hidden_size": 768,
        "sampling_rate": 16_000,
        "normalize_input": False,
    },
}


class SSLFrontendConfig(PretrainedConfig):
    """Configuration for a supported Hugging Face SSL speech frontend."""

    model_type = "ssl_frontend"

    def __init__(
        self,
        ssl_type: str = "wav2vec2-base",
        target_sampling_rate: int = 16_000,
        output_size: int = 768,
        input_norm_eps: float = 1e-7,
        freeze_ssl_model: bool = False,
        ssl_trainable_layer_indices: Optional[list[int]] = None,
        ssl_model_config: Optional[dict[str, object]] = None,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if ssl_type not in SSL_FRONTEND_SPECS:
            supported = ", ".join(SSL_FRONTEND_SPECS)
            raise ValueError(
                f"Unsupported ssl_type={ssl_type!r}; expected one of {supported}."
            )
        spec = SSL_FRONTEND_SPECS[ssl_type]
        expected_sampling_rate = int(spec["sampling_rate"])
        if target_sampling_rate != expected_sampling_rate:
            raise ValueError(
                f"{ssl_type} expects target_sampling_rate={expected_sampling_rate}, "
                f"got {target_sampling_rate}."
            )
        if output_size <= 0:
            raise ValueError("output_size must be positive.")
        if input_norm_eps <= 0:
            raise ValueError("input_norm_eps must be positive.")
        if not isinstance(freeze_ssl_model, bool):
            raise TypeError("freeze_ssl_model must be a bool.")
        if ssl_trainable_layer_indices is not None:
            if not freeze_ssl_model:
                raise ValueError(
                    "ssl_trainable_layer_indices requires freeze_ssl_model=True."
                )
            if not isinstance(ssl_trainable_layer_indices, list):
                raise TypeError("ssl_trainable_layer_indices must be a list or None.")
            if not ssl_trainable_layer_indices:
                raise ValueError("ssl_trainable_layer_indices must not be empty.")
            if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in ssl_trainable_layer_indices
            ):
                raise TypeError("ssl_trainable_layer_indices must contain only integers.")
            if any(index < 0 for index in ssl_trainable_layer_indices):
                raise ValueError("ssl_trainable_layer_indices must be non-negative.")
            if len(set(ssl_trainable_layer_indices)) != len(ssl_trainable_layer_indices):
                raise ValueError("ssl_trainable_layer_indices must not contain duplicates.")
        if ssl_model_config is not None and not isinstance(ssl_model_config, dict):
            raise TypeError("ssl_model_config must be a dict or None.")
        if initializer_range <= 0:
            raise ValueError("initializer_range must be positive.")

        self.ssl_type = ssl_type
        self.pretrained_model_name_or_path = str(spec["pretrained_model_name_or_path"])
        self.target_sampling_rate = target_sampling_rate
        self.ssl_hidden_size = int(spec["hidden_size"])
        self.output_size = output_size
        self.normalize_input = bool(spec["normalize_input"])
        self.input_norm_eps = input_norm_eps
        self.freeze_ssl_model = freeze_ssl_model
        self.ssl_trainable_layer_indices = (
            list(ssl_trainable_layer_indices)
            if ssl_trainable_layer_indices is not None
            else None
        )
        self.ssl_model_config = (
            dict(ssl_model_config) if ssl_model_config is not None else None
        )
        self.initializer_range = initializer_range

    @property
    def frontend_output_dim(self) -> int:
        return self.output_size


class RavenFLAConfig(PretrainedConfig):
    """Configuration for the FLA Raven sequence mixer."""

    model_type = "raven_fla"
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        num_heads: int = 4,
        num_kv_heads: int | None = 2,
        num_slots: int | None = None, # 内部计算
        topk: int = 32, # 必须小于 num_slots
        feature_map: str = "swish",
        decay_type: str = "Mamba2", # or 'GLA'
        router_score: str = "sigmoid", # or 'softmax'
        router_type: str = "lin", # or 'mlp'
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_slots = num_slots
        self.topk = topk
        self.feature_map = feature_map
        self.decay_type = decay_type
        self.router_score = router_score
        self.router_type = router_type


class GatedDelta2FLAConfig(PretrainedConfig):
    """Configuration for the FLA GatedDeltaNet2 sequence mixer."""

    model_type = "gated_delta2_fla"
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        num_heads: int = 4,
        num_v_heads: int | None = 8, # 整除 num_heads, 相当于 n 个 v 共享一组 qk
        head_dim: int = 64, # 默认 256
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.head_dim = head_dim

class Mamba3FLAConfig(PretrainedConfig):
    """Configuration for the FLA Mamba3 sequence mixer."""

    model_type = "mamba3_fla"
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        head_dim: int = 64, # 默认 expand=2，内部头数 = 2 * hidden_size // head_dim
        state_size: int = 128, # 64 or 128
        is_mimo: bool = False, # Mamba3 特有，是否启用 MIMO 版本，否则 SISO
        rescale_prenorm_residual: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(rescale_prenorm_residual, bool):
            raise TypeError("rescale_prenorm_residual must be a bool.")
        self.state_size = state_size
        self.head_dim = head_dim
        self.is_mimo = is_mimo
        self.rescale_prenorm_residual = rescale_prenorm_residual



class AttentionFLAConfig(PretrainedConfig):
    """Configuration for the FLA standard Attention mixer."""

    model_type = "attention_fla"
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        num_heads: int = 4,
        num_kv_heads: int | None = 2,
        qk_norm: bool = False,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.0,
        max_position_embeddings: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings


class AttentionBackboneConfig(PretrainedConfig):
    """Configuration for the pre-norm FLA/attention + MLP/MoE backbone."""

    model_type = "attention_backbone"
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 4, # Raven，GDN2，Attention 类使用，公共参数可不使用
        num_kv_heads: Optional[int] = None, # Attention 类使用
        blocks: Optional[Sequence[Any] | dict[Any, Any]] = None,
        hrm_h_blocks: Optional[Sequence[Any] | dict[Any, Any]] = None,
        hrm_l_blocks: Optional[Sequence[Any] | dict[Any, Any]] = None,
        bias: bool = False, # 常关
        fuse_norm: bool = True,

        # Backbone 架构配置
        architecture_type: str = "baseline", # baseline / hrm / heterogeneous_hrm / looped
        looped_num_cycles: int = 2, # N: 同一组 backbone 参数的总循环次数
        looped_num_gradient_cycles: int = 1, # K: 最后参与反向传播的循环次数
        hrm_h_cycles: int = 2, # H2L3 中的 H
        hrm_l_cycles: int = 3, # H2L3 中的 L
        hrm_num_gradient_steps: int = 2, # K: 末尾连续参与反向传播的完整 module 次数

        # MLP Dense 层配置
        mlp_hidden_ratio: Optional[int] = None, # 中间层缩放系数，默认为4，一般不使用
        mlp_intermediate_size: Optional[int] = None, # 手动指定，就不会使用 ratio，共享专家和普通 ffn 都使用这个中间维度
        mlp_hidden_act: str = "swish", # 仅支持 swish
        mlp_fuse_swiglu: bool = True,
        residual_dropout: float = 0.0, # mixer 与 FFN/MoE 输出在残差相加前使用

        # MoE 配置
        moe_latent_hidden_size: Optional[int] = 128, # [hidden_size -> latent_hidden_size -> intermediate_size], 256 -> 128 降维
        moe_latent_intermediate_size: Optional[int] = 192, # 128 -> 192
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_capacity_factor: Optional[float] = None, # drop_tokens 不使用
        moe_drop_tokens: bool = False, # drop_tokens 不使用
        moe_aux_loss_coeff: float = 0.0, # MoE 辅助损失，让专家更均衡
        moe_use_seq_aux_loss: bool = False, # 启用损失值的情况下，seq 关闭则计算 global
        moe_use_shared_expert: bool = False, # 共享专家
        return_aux_loss: bool = False, # 需要与 moe_aux_loss_coeff 配合使用，返回辅助损失值

        # Block 配置
        use_cache: bool = False, # 全局默认关闭；需要缓存时显式开启
        norm_eps: float = 1e-6,
        attnres_block_size: Optional[int] = None, # 不使用

        # 嵌套配置，方便调用
        attention_config: Optional[AttentionFLAConfig] = None,
        raven_config: Optional[RavenFLAConfig] = None,
        gdn2_config: Optional[GatedDelta2FLAConfig] = None,
        mamba3_config: Optional[Mamba3FLAConfig] = None,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        if architecture_type not in {"baseline", "hrm", "heterogeneous_hrm", "looped"}:
            raise ValueError(
                "architecture_type must be one of baseline, hrm, heterogeneous_hrm, or looped, "
                f"got {architecture_type!r}."
            )
        if architecture_type == "heterogeneous_hrm":
            if blocks is not None:
                raise ValueError(
                    "heterogeneous_hrm uses hrm_h_blocks and hrm_l_blocks; "
                    "do not also provide blocks."
                )
            if hrm_h_blocks is None or hrm_l_blocks is None:
                raise ValueError(
                    "heterogeneous_hrm requires both hrm_h_blocks and hrm_l_blocks."
                )
            self.blocks = None
            self.hrm_h_blocks = self._normalize_blocks(hrm_h_blocks)
            self.hrm_l_blocks = self._normalize_blocks(hrm_l_blocks)
            if self.hrm_h_num_hidden_layers != self.hrm_l_num_hidden_layers:
                raise ValueError(
                    "heterogeneous_hrm currently requires H and L to have the same number "
                    f"of layers, got H={self.hrm_h_num_hidden_layers} and "
                    f"L={self.hrm_l_num_hidden_layers}."
                )
        else:
            if hrm_h_blocks is not None or hrm_l_blocks is not None:
                raise ValueError(
                    "hrm_h_blocks and hrm_l_blocks are only valid for "
                    "architecture_type='heterogeneous_hrm'."
                )
            self.blocks = self._normalize_blocks(blocks)
            self.hrm_h_blocks = None
            self.hrm_l_blocks = None
        if isinstance(looped_num_cycles, bool) or not isinstance(looped_num_cycles, int):
            raise TypeError("looped_num_cycles must be an integer.")
        if isinstance(looped_num_gradient_cycles, bool) or not isinstance(
            looped_num_gradient_cycles,
            int,
        ):
            raise TypeError("looped_num_gradient_cycles must be an integer.")
        if looped_num_cycles <= 0:
            raise ValueError("looped_num_cycles must be positive.")
        if not 1 <= looped_num_gradient_cycles <= looped_num_cycles:
            raise ValueError(
                "looped_num_gradient_cycles must satisfy "
                "1 <= K <= looped_num_cycles."
            )
        for name, value in (
            ("hrm_h_cycles", hrm_h_cycles),
            ("hrm_l_cycles", hrm_l_cycles),
            ("hrm_num_gradient_steps", hrm_num_gradient_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        hrm_total_steps = hrm_h_cycles * (hrm_l_cycles + 1)
        if hrm_num_gradient_steps > hrm_total_steps:
            raise ValueError(
                "hrm_num_gradient_steps must satisfy "
                f"1 <= K <= H*(L+1)={hrm_total_steps}."
            )
        self.architecture_type = architecture_type
        self.looped_num_cycles = looped_num_cycles
        self.looped_num_gradient_cycles = looped_num_gradient_cycles
        self.hrm_h_cycles = hrm_h_cycles
        self.hrm_l_cycles = hrm_l_cycles
        self.hrm_num_gradient_steps = hrm_num_gradient_steps
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError("residual_dropout must satisfy 0.0 <= residual_dropout < 1.0.")
        self.residual_dropout = residual_dropout
        self.bias = bias
        self.fuse_norm = fuse_norm
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.mlp_intermediate_size = mlp_intermediate_size
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_fuse_swiglu = mlp_fuse_swiglu
        self.moe_latent_hidden_size = moe_latent_hidden_size
        self.moe_latent_intermediate_size = moe_latent_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_drop_tokens = moe_drop_tokens
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.moe_use_seq_aux_loss = moe_use_seq_aux_loss
        self.moe_use_shared_expert = moe_use_shared_expert
        self.return_aux_loss = return_aux_loss
        self.use_cache = use_cache
        self.norm_eps = norm_eps
        self.attnres_block_size = attnres_block_size
        self.attention_config = _config_from_dict(attention_config, (AttentionFLAConfig,)) if attention_config is not None else AttentionFLAConfig()
        self.raven_config = _config_from_dict(raven_config, (RavenFLAConfig,)) if raven_config is not None else RavenFLAConfig()
        self.gdn2_config = _config_from_dict(gdn2_config, (GatedDelta2FLAConfig,)) if gdn2_config is not None else GatedDelta2FLAConfig()
        self.mamba3_config = _config_from_dict(mamba3_config, (Mamba3FLAConfig,)) if mamba3_config is not None else Mamba3FLAConfig()
        if not isinstance(self.attention_config, AttentionFLAConfig):
            raise TypeError("attention_config must be an AttentionFLAConfig or None.")
        if not isinstance(self.raven_config, RavenFLAConfig):
            raise TypeError("raven_config must be a RavenFLAConfig or None.")
        if not isinstance(self.gdn2_config, GatedDelta2FLAConfig):
            raise TypeError("gdn2_config must be a GatedDelta2FLAConfig or None.")
        if not isinstance(self.mamba3_config, Mamba3FLAConfig):
            raise TypeError("mamba3_config must be a Mamba3FLAConfig or None.")
        self.initializer_range = initializer_range

    @staticmethod
    def _default_blocks() -> list[dict[str, object]]:
        return [
            {"attn": "raven", "mlp": "mlp", "num_layers": 1},
            {"attn": "raven", "mlp": "moe", "num_layers": 1},
            {"attn": "attention", "mlp": "moe", "num_layers": 1},
        ]

    @classmethod
    def _normalize_blocks(
        cls,
        blocks: Optional[Sequence[Any] | dict[Any, Any]],
    ) -> list[dict[str, object]]:
        if blocks is None:
            entries = list(enumerate(cls._default_blocks()))
        elif isinstance(blocks, dict):
            entries = list(blocks.items())
            if all(cls._is_int_like(key) for key, _ in entries):
                entries.sort(key=lambda item: int(item[0]))
        else:
            entries = list(enumerate(blocks))

        normalized = [cls._normalize_block_spec(key, spec) for key, spec in entries]
        if not normalized:
            raise ValueError("blocks must contain at least one block specification.")
        return normalized

    @staticmethod
    def _is_int_like(value: object) -> bool:
        try:
            int(value)
        except (TypeError, ValueError):
            return False
        return True

    @classmethod
    def _normalize_block_spec(cls, key: object, spec: Any) -> dict[str, object]:
        if isinstance(spec, dict):
            attn = spec.get("attn")
            mlp = spec.get("mlp")
            num_layers = spec.get("num_layers", 1)
        elif isinstance(spec, (list, tuple)):
            if len(spec) != 3:
                raise ValueError(
                    f"blocks[{key!r}] must have three entries: attn, mlp, num_layers."
                )
            attn, mlp, num_layers = spec
        else:
            raise TypeError(
                f"blocks[{key!r}] must be a dict or a three-item tuple/list, got {type(spec).__name__}."
            )

        if attn is None or mlp is None:
            raise ValueError(f"blocks[{key!r}] must define both attn and mlp.")
        try:
            layer_count = int(num_layers)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"blocks[{key!r}].num_layers must be an integer.") from exc
        if layer_count <= 0:
            raise ValueError(f"blocks[{key!r}].num_layers must be positive.")
        if attn not in {"raven", "gdn2", "mamba3", "attention"}:
            raise ValueError(
                f"blocks[{key!r}].attn must be one of raven, gdn2, mamba3, or attention; got {attn!r}."
            )
        if mlp not in {"mlp", "moe"}:
            raise ValueError(f"blocks[{key!r}].mlp must be one of mlp or moe; got {mlp!r}.")

        return {
            "attn": attn,
            "mlp": mlp,
            "num_layers": layer_count,
        }

    @property
    def resolved_hidden_size(self) -> int:
        return self.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        if self.architecture_type == "heterogeneous_hrm":
            return self.hrm_h_num_hidden_layers
        assert self.blocks is not None
        return sum(int(spec["num_layers"]) for spec in self.blocks)

    @property
    def hrm_h_num_hidden_layers(self) -> int:
        if self.hrm_h_blocks is None:
            return 0
        return sum(int(spec["num_layers"]) for spec in self.hrm_h_blocks)

    @property
    def hrm_l_num_hidden_layers(self) -> int:
        if self.hrm_l_blocks is None:
            return 0
        return sum(int(spec["num_layers"]) for spec in self.hrm_l_blocks)

    @property
    def resolved_moe_latent_hidden_size(self) -> int:
        return self.moe_latent_hidden_size or self.resolved_hidden_size

    @property
    def resolved_moe_latent_intermediate_size(self) -> int:
        return self.moe_latent_intermediate_size or 4 * self.resolved_moe_latent_hidden_size


class AASISTConfig(PretrainedConfig):
    """Configuration for the SSL feature to AASIST embedding backend."""

    model_type = "aasist"

    def __init__(
        self,
        input_size: int = 768,
        filts: Optional[Sequence[Any]] = None,
        gat_dims: Optional[Sequence[int]] = None,
        pool_ratios: Optional[Sequence[float]] = None,
        temperatures: Optional[Sequence[float]] = None,
        graph_attention_dropout: float = 0.2,
        graph_pool_dropout: float = 0.3,
        path_dropout: float = 0.2,
        output_dropout: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(input_size, bool) or not isinstance(input_size, int):
            raise TypeError("input_size must be an integer.")
        if input_size <= 0:
            raise ValueError("input_size must be positive.")

        resolved_filts = list(filts) if filts is not None else [
            128,
            [1, 32],
            [32, 32],
            [32, 64],
            [64, 64],
        ]
        if len(resolved_filts) != 5:
            raise ValueError("filts must contain one feature size and four channel pairs.")
        feature_size = resolved_filts[0]
        if isinstance(feature_size, bool) or not isinstance(feature_size, int):
            raise TypeError("filts[0] must be an integer.")
        if feature_size < 3:
            raise ValueError("filts[0] must be at least 3 for the initial max pooling.")

        channel_pairs: list[list[int]] = []
        for index, pair in enumerate(resolved_filts[1:], start=1):
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise TypeError(f"filts[{index}] must be a two-element channel pair.")
            normalized_pair = list(pair)
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in normalized_pair
            ):
                raise ValueError(f"filts[{index}] channel values must be positive integers.")
            channel_pairs.append(normalized_pair)
        if channel_pairs[0][0] != 1:
            raise ValueError("filts[1][0] must be 1 because AASIST starts from one 2D channel.")
        for previous, current in zip(channel_pairs, channel_pairs[1:]):
            if previous[1] != current[0]:
                raise ValueError(
                    "Adjacent filts channel pairs must be continuous, got "
                    f"{previous} followed by {current}."
                )
        if channel_pairs[-1][0] != channel_pairs[-1][1]:
            raise ValueError(
                "The final filts channel pair must keep the same channel size because "
                "that residual block is repeated three times."
            )

        resolved_gat_dims = list(gat_dims) if gat_dims is not None else [64, 32]
        if len(resolved_gat_dims) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolved_gat_dims
        ):
            raise ValueError("gat_dims must contain two positive integers.")

        resolved_pool_ratios = list(pool_ratios) if pool_ratios is not None else [
            0.5,
            0.5,
            0.5,
            0.5,
        ]
        if len(resolved_pool_ratios) != 4 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 1.0
            for value in resolved_pool_ratios
        ):
            raise ValueError("pool_ratios must contain four values in the interval (0, 1].")

        resolved_temperatures = list(temperatures) if temperatures is not None else [
            2.0,
            2.0,
            100.0,
            100.0,
        ]
        if len(resolved_temperatures) != 4 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) <= 0.0
            for value in resolved_temperatures
        ):
            raise ValueError("temperatures must contain four positive values.")

        for name, value in (
            ("graph_attention_dropout", graph_attention_dropout),
            ("graph_pool_dropout", graph_pool_dropout),
            ("path_dropout", path_dropout),
            ("output_dropout", output_dropout),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a float.")
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must satisfy 0.0 <= {name} < 1.0.")

        self.input_size = input_size
        self.filts = [feature_size, *channel_pairs]
        self.gat_dims = resolved_gat_dims
        self.pool_ratios = [float(value) for value in resolved_pool_ratios]
        self.temperatures = [float(value) for value in resolved_temperatures]
        self.graph_attention_dropout = float(graph_attention_dropout)
        self.graph_pool_dropout = float(graph_pool_dropout)
        self.path_dropout = float(path_dropout)
        self.output_dropout = float(output_dropout)

    @property
    def feature_size(self) -> int:
        return int(self.filts[0])

    @property
    def encoder_output_size(self) -> int:
        return int(self.filts[-1][-1])

    @property
    def spectral_num_nodes(self) -> int:
        return self.feature_size // 3

    @property
    def output_size(self) -> int:
        return 5 * int(self.gat_dims[1])


class ADDConfig(PretrainedConfig):
    """Configuration for selectable ADD audio-classification pipelines."""

    model_type = "add"

    def __init__(
        self,
        frontend_config: SincNetFrontendConfig | LinearFrontendConfig | SSLFrontendConfig,
        backbone_config: AttentionBackboneConfig | None,
        pooling_config: GatedAttentionPoolingConfig | None,
        classifier_config: LinearClassifierConfig,
        aasist_config: AASISTConfig | None = None,
        model_architecture: str = "backbone_pooling",
        architecture_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        frontend_config = _config_from_dict(
            frontend_config,
            (SincNetFrontendConfig, LinearFrontendConfig, SSLFrontendConfig),
        )
        backbone_config = (
            _config_from_dict(backbone_config, (AttentionBackboneConfig,))
            if backbone_config is not None
            else None
        )
        pooling_config = (
            _config_from_dict(pooling_config, (GatedAttentionPoolingConfig,))
            if pooling_config is not None
            else None
        )
        aasist_config = (
            _config_from_dict(aasist_config, (AASISTConfig,))
            if aasist_config is not None
            else None
        )
        classifier_config = _config_from_dict(classifier_config, (LinearClassifierConfig,))
        if not isinstance(
            frontend_config,
            (SincNetFrontendConfig, LinearFrontendConfig, SSLFrontendConfig),
        ):
            raise TypeError(
                "frontend_config must be a SincNetFrontendConfig, "
                "LinearFrontendConfig, or SSLFrontendConfig."
            )
        if not isinstance(classifier_config, LinearClassifierConfig):
            raise TypeError("classifier_config must be a LinearClassifierConfig.")
        if model_architecture not in {"backbone_pooling", "ssl_aasist"}:
            raise ValueError(
                "model_architecture must be one of backbone_pooling or ssl_aasist, "
                f"got {model_architecture!r}."
            )

        if model_architecture == "backbone_pooling":
            if not isinstance(backbone_config, AttentionBackboneConfig):
                raise TypeError(
                    "model_architecture='backbone_pooling' requires an "
                    "AttentionBackboneConfig."
                )
            if not isinstance(pooling_config, GatedAttentionPoolingConfig):
                raise TypeError(
                    "model_architecture='backbone_pooling' requires a "
                    "GatedAttentionPoolingConfig."
                )
            if aasist_config is not None:
                raise ValueError(
                    "aasist_config must be None for model_architecture='backbone_pooling'."
                )
            if (
                architecture_type is not None
                and architecture_type != backbone_config.architecture_type
            ):
                raise ValueError(
                    "ADDConfig architecture_type must match backbone_config.architecture_type: "
                    f"{architecture_type!r} != {backbone_config.architecture_type!r}."
                )
            resolved_architecture_type = backbone_config.architecture_type
        else:
            if not isinstance(frontend_config, SSLFrontendConfig):
                raise TypeError(
                    "model_architecture='ssl_aasist' requires an SSLFrontendConfig."
                )
            if not isinstance(aasist_config, AASISTConfig):
                raise TypeError(
                    "model_architecture='ssl_aasist' requires an AASISTConfig."
                )
            if backbone_config is not None or pooling_config is not None:
                raise ValueError(
                    "backbone_config and pooling_config must be None for "
                    "model_architecture='ssl_aasist'."
                )
            if architecture_type not in {None, "ssl_aasist"}:
                raise ValueError(
                    "architecture_type must be None or 'ssl_aasist' when "
                    "model_architecture='ssl_aasist'."
                )
            resolved_architecture_type = "ssl_aasist"

        self.model_architecture = model_architecture
        self.architecture_type = resolved_architecture_type
        self.frontend_config = frontend_config
        self.backbone_config = backbone_config
        self.pooling_config = pooling_config
        self.aasist_config = aasist_config
        self.classifier_config = classifier_config


class GatedAttentionPoolingConfig(PretrainedConfig):
    """Configuration shared by the selectable pooling implementations."""

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
        gate_act: str = "silu", # sigmoid or silu
        score_temperature: float = 1.0, # < 1.0 尖锐，选择性更强；> 1.0 平滑，类似加权平均
        hidden_ratio: float = 2.0, # gapv1 / gapv2 使用
        output_act: str = "none",
        use_output_norm: bool = False,
        layer_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Normalize legacy pooling-family model types when loading old checkpoints.
        self.model_type = type(self).model_type
        if not isinstance(pooling_type, str) or not pooling_type.strip():
            raise TypeError("pooling_type must be a non-empty string.")
        self.pooling_type = pooling_type.strip().lower()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if output_size <= 0:
            raise ValueError("output_size must be positive.")
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0.")
        self.dropout = dropout
        self.bias = bias
        if gate_act not in {"sigmoid", "silu"}:
            raise ValueError(f"gate_act must be one of sigmoid or silu, got {gate_act!r}.")
        self.gate_act = gate_act
        if isinstance(score_temperature, bool) or not isinstance(score_temperature, (int, float)):
            raise TypeError("score_temperature must be a float.")
        if score_temperature <= 0:
            raise ValueError("score_temperature must be positive.")
        self.score_temperature = float(score_temperature)
        if isinstance(hidden_ratio, bool) or not isinstance(hidden_ratio, (int, float)):
            raise TypeError("hidden_ratio must be a float.")
        if hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be positive.")
        self.hidden_ratio = float(hidden_ratio)
        if output_act not in {"none", "silu"}:
            raise ValueError(
                f"output_act must be one of none or silu, got {output_act!r}."
            )
        self.output_act = output_act
        if not isinstance(use_output_norm, bool):
            raise TypeError("use_output_norm must be a bool.")
        self.use_output_norm = use_output_norm
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

    @property
    def resolved_num_heads(self) -> int:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")
        if self.output_size <= 0:
            raise ValueError("output_size must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.output_size % self.num_heads != 0:
            raise ValueError(
                f"output_size={self.output_size} must be divisible by num_heads={self.num_heads}."
            )
        return self.num_heads

    @property
    def resolved_head_dim(self) -> int:
        return self.output_size // self.resolved_num_heads


class LinearClassifierConfig(PretrainedConfig):
    """Configuration for the classifier head."""

    model_type = "linear_classifier"

    def __init__(
        self,
        input_size: int = 128,
        num_labels: int = 2,
        classifier_type: str = "linear",
        bias: bool = False,
        am_margin: float = 0.3,
        am_scale: float = 15.0,
        am_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.input_size = input_size
        self.num_labels = num_labels
        if classifier_type not in {"linear", "amsoftmax"}:
            raise ValueError(f"classifier_type must be one of linear or amsoftmax, got {classifier_type!r}.")
        if am_margin < 0:
            raise ValueError("am_margin must be non-negative.")
        if am_scale <= 0:
            raise ValueError("am_scale must be positive.")
        if am_eps <= 0:
            raise ValueError("am_eps must be positive.")
        self.classifier_type = classifier_type
        self.bias = bias
        self.am_margin = am_margin
        self.am_scale = am_scale
        self.am_eps = am_eps
        self.initializer_range = initializer_range
