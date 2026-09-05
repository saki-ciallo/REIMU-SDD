from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from transformers import PretrainedConfig

from ..blocks import (
    ArchitectureType,
    BlockSpec,
    MixerType,
    RawBlockCollection,
    legacy_block_dicts,
    normalize_block_specs,
)
from ..validation import (
    boolean,
    coerce_config,
    optional_positive_int,
    positive_float,
    positive_int,
    probability,
)
from .mixers import (
    AttentionFLAConfig,
    GatedDelta2FLAConfig,
    Mamba3FLAConfig,
    RavenFLAConfig,
)

DEFAULT_BLOCKS: tuple[dict[str, object], ...] = (
    {"attn": "raven", "mlp": "mlp", "num_layers": 1},
    {"attn": "raven", "mlp": "moe", "num_layers": 1},
    {"attn": "attention", "mlp": "moe", "num_layers": 1},
)


def _architecture_type(value: object) -> ArchitectureType:
    if not isinstance(value, str):
        raise TypeError(f"architecture_type must be a string, got {type(value).__name__}.")
    try:
        return ArchitectureType(value.strip().lower())
    except ValueError as exc:
        supported = ", ".join(member.value for member in ArchitectureType)
        raise ValueError(f"architecture_type must be one of {supported}, got {value!r}.") from exc


def _count_layers(specs: Sequence[BlockSpec] | None) -> int:
    return sum(spec.num_layers for spec in specs or ())


class AttentionBackboneConfig(PretrainedConfig):
    """Validated configuration shared by all four backbone architectures."""

    model_type = "attention_backbone"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 512,
        blocks: RawBlockCollection | None = None,
        hrm_h_blocks: RawBlockCollection | None = None,
        hrm_l_blocks: RawBlockCollection | None = None,
        bias: bool = False,
        fuse_norm: bool = True,
        architecture_type: str = "baseline",
        looped_num_cycles: int = 2,
        looped_num_gradient_cycles: int = 1,
        hrm_h_cycles: int = 2,
        hrm_l_cycles: int = 3,
        hrm_num_gradient_steps: int = 2,
        mlp_hidden_ratio: float | None = None,
        mlp_intermediate_size: int | None = None,
        mlp_hidden_act: str = "swish",
        mlp_fuse_swiglu: bool = True,
        residual_dropout: float = 0.0,
        moe_latent_hidden_size: int | None = 128,
        moe_latent_intermediate_size: int | None = 192,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_capacity_factor: float | None = None,
        moe_drop_tokens: bool = False,
        moe_aux_loss_coeff: float = 0.0,
        moe_use_seq_aux_loss: bool = False,
        moe_use_shared_expert: bool = False,
        return_aux_loss: bool = False,
        use_cache: bool = False,
        norm_eps: float = 1e-6,
        attnres_block_size: int | None = None,
        attention_config: AttentionFLAConfig | Mapping[str, Any] | None = None,
        raven_config: RavenFLAConfig | Mapping[str, Any] | None = None,
        gdn2_config: GatedDelta2FLAConfig | Mapping[str, Any] | None = None,
        mamba3_config: Mamba3FLAConfig | Mapping[str, Any] | None = None,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        architecture = _architecture_type(architecture_type)
        self.hidden_size = positive_int(hidden_size, field_name="hidden_size")
        self.bias = boolean(bias, field_name="bias")
        self.fuse_norm = boolean(fuse_norm, field_name="fuse_norm")

        if architecture is ArchitectureType.HETEROGENEOUS_HRM:
            if blocks is not None:
                raise ValueError(
                    "heterogeneous_hrm uses hrm_h_blocks and hrm_l_blocks; blocks must be omitted."
                )
            if hrm_h_blocks is None or hrm_l_blocks is None:
                raise ValueError("heterogeneous_hrm requires hrm_h_blocks and hrm_l_blocks.")
            block_specs = None
            h_specs = normalize_block_specs(hrm_h_blocks)
            l_specs = normalize_block_specs(hrm_l_blocks)
            if _count_layers(h_specs) != _count_layers(l_specs):
                raise ValueError(
                    "heterogeneous_hrm requires H and L to have equal depths; "
                    f"got H={_count_layers(h_specs)} and L={_count_layers(l_specs)}."
                )
        else:
            if hrm_h_blocks is not None or hrm_l_blocks is not None:
                raise ValueError(
                    "hrm_h_blocks and hrm_l_blocks are only valid for "
                    "architecture_type='heterogeneous_hrm'."
                )
            block_specs = normalize_block_specs(blocks or DEFAULT_BLOCKS)
            h_specs = None
            l_specs = None

        self.architecture_type = architecture.value
        self.blocks = legacy_block_dicts(block_specs) if block_specs is not None else None
        self.hrm_h_blocks = legacy_block_dicts(h_specs) if h_specs is not None else None
        self.hrm_l_blocks = legacy_block_dicts(l_specs) if l_specs is not None else None

        self.looped_num_cycles = positive_int(
            looped_num_cycles,
            field_name="looped_num_cycles",
        )
        self.looped_num_gradient_cycles = positive_int(
            looped_num_gradient_cycles,
            field_name="looped_num_gradient_cycles",
        )
        if self.looped_num_gradient_cycles > self.looped_num_cycles:
            raise ValueError("looped_num_gradient_cycles must not exceed looped_num_cycles.")

        self.hrm_h_cycles = positive_int(hrm_h_cycles, field_name="hrm_h_cycles")
        self.hrm_l_cycles = positive_int(hrm_l_cycles, field_name="hrm_l_cycles")
        self.hrm_num_gradient_steps = positive_int(
            hrm_num_gradient_steps,
            field_name="hrm_num_gradient_steps",
        )
        if self.hrm_num_gradient_steps > self.hrm_total_steps:
            raise ValueError(
                f"hrm_num_gradient_steps must not exceed H*(L+1)={self.hrm_total_steps}."
            )

        self.mlp_hidden_ratio = (
            positive_float(mlp_hidden_ratio, field_name="mlp_hidden_ratio")
            if mlp_hidden_ratio is not None
            else None
        )
        self.mlp_intermediate_size = optional_positive_int(
            mlp_intermediate_size,
            field_name="mlp_intermediate_size",
        )
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_fuse_swiglu = boolean(
            mlp_fuse_swiglu,
            field_name="mlp_fuse_swiglu",
        )
        self.residual_dropout = probability(
            residual_dropout,
            field_name="residual_dropout",
        )

        self.moe_latent_hidden_size = optional_positive_int(
            moe_latent_hidden_size,
            field_name="moe_latent_hidden_size",
        )
        self.moe_latent_intermediate_size = optional_positive_int(
            moe_latent_intermediate_size,
            field_name="moe_latent_intermediate_size",
        )
        self.moe_num_experts = positive_int(
            moe_num_experts,
            field_name="moe_num_experts",
        )
        self.moe_top_k = positive_int(moe_top_k, field_name="moe_top_k")
        if self.moe_top_k > self.moe_num_experts:
            raise ValueError("moe_top_k must not exceed moe_num_experts.")
        self.moe_capacity_factor = (
            positive_float(moe_capacity_factor, field_name="moe_capacity_factor")
            if moe_capacity_factor is not None
            else None
        )
        self.moe_drop_tokens = boolean(
            moe_drop_tokens,
            field_name="moe_drop_tokens",
        )
        if self.moe_drop_tokens and self.moe_capacity_factor is None:
            raise ValueError("moe_drop_tokens=True requires moe_capacity_factor.")
        if isinstance(moe_aux_loss_coeff, bool) or not isinstance(
            moe_aux_loss_coeff,
            (int, float),
        ):
            raise TypeError("moe_aux_loss_coeff must be a number.")
        if moe_aux_loss_coeff < 0:
            raise ValueError("moe_aux_loss_coeff must be non-negative.")
        self.moe_aux_loss_coeff = float(moe_aux_loss_coeff)
        self.moe_use_seq_aux_loss = boolean(
            moe_use_seq_aux_loss,
            field_name="moe_use_seq_aux_loss",
        )
        self.moe_use_shared_expert = boolean(
            moe_use_shared_expert,
            field_name="moe_use_shared_expert",
        )
        self.return_aux_loss = boolean(return_aux_loss, field_name="return_aux_loss")

        self.use_cache = boolean(use_cache, field_name="use_cache")
        self.norm_eps = positive_float(norm_eps, field_name="norm_eps")
        self.attnres_block_size = optional_positive_int(
            attnres_block_size,
            field_name="attnres_block_size",
        )
        self.initializer_range = positive_float(
            initializer_range,
            field_name="initializer_range",
        )

        self.attention_config = coerce_config(
            attention_config,
            AttentionFLAConfig,
            field_name="attention_config",
            default=True,
        )
        self.raven_config = coerce_config(
            raven_config,
            RavenFLAConfig,
            field_name="raven_config",
            default=True,
        )
        self.gdn2_config = coerce_config(
            gdn2_config,
            GatedDelta2FLAConfig,
            field_name="gdn2_config",
            default=True,
        )
        self.mamba3_config = coerce_config(
            mamba3_config,
            Mamba3FLAConfig,
            field_name="mamba3_config",
            default=True,
        )
        self._validate_selected_mixer_dimensions()

    @property
    def block_specs(self) -> tuple[BlockSpec, ...]:
        return normalize_block_specs(self.blocks) if self.blocks is not None else ()

    @property
    def hrm_h_block_specs(self) -> tuple[BlockSpec, ...]:
        return normalize_block_specs(self.hrm_h_blocks) if self.hrm_h_blocks is not None else ()

    @property
    def hrm_l_block_specs(self) -> tuple[BlockSpec, ...]:
        return normalize_block_specs(self.hrm_l_blocks) if self.hrm_l_blocks is not None else ()

    @property
    def all_block_specs(self) -> tuple[BlockSpec, ...]:
        return (*self.block_specs, *self.hrm_h_block_specs, *self.hrm_l_block_specs)

    @property
    def resolved_hidden_size(self) -> int:
        return self.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        if self.architecture_type == ArchitectureType.HETEROGENEOUS_HRM:
            return self.hrm_h_num_hidden_layers
        return _count_layers(self.block_specs)

    @property
    def hrm_h_num_hidden_layers(self) -> int:
        return _count_layers(self.hrm_h_block_specs)

    @property
    def hrm_l_num_hidden_layers(self) -> int:
        return _count_layers(self.hrm_l_block_specs)

    @property
    def hrm_total_steps(self) -> int:
        return self.hrm_h_cycles * (self.hrm_l_cycles + 1)

    @property
    def looped_gradient_schedule(self) -> tuple[bool, ...]:
        frozen_cycles = self.looped_num_cycles - self.looped_num_gradient_cycles
        return (False,) * frozen_cycles + (True,) * self.looped_num_gradient_cycles

    @property
    def hrm_gradient_schedule(self) -> tuple[bool, ...]:
        frozen_steps = self.hrm_total_steps - self.hrm_num_gradient_steps
        return (False,) * frozen_steps + (True,) * self.hrm_num_gradient_steps

    @property
    def resolved_moe_latent_hidden_size(self) -> int:
        return self.moe_latent_hidden_size or self.hidden_size

    @property
    def resolved_moe_latent_intermediate_size(self) -> int:
        return self.moe_latent_intermediate_size or 4 * self.resolved_moe_latent_hidden_size

    def _validate_selected_mixer_dimensions(self) -> None:
        selected = {spec.mixer for spec in self.all_block_specs}
        if MixerType.ATTENTION in selected and self.hidden_size % self.attention_config.num_heads:
            raise ValueError("hidden_size must be divisible by attention_config.num_heads.")
        if MixerType.RAVEN in selected and self.hidden_size % self.raven_config.num_heads:
            raise ValueError("hidden_size must be divisible by raven_config.num_heads.")
        if MixerType.MAMBA3 in selected:
            expanded_size = 2 * self.hidden_size
            if expanded_size % self.mamba3_config.head_dim:
                raise ValueError("2 * hidden_size must be divisible by mamba3_config.head_dim.")
