from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from torch import nn
from fla.layers import GatedDeltaNet2, Raven, Mamba3
from fla.layers.attn import Attention
from fla.models.utils import Cache
from fla.modules import GatedMLP, RMSNorm
from fla.ops.attnres import fused_attnres
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .backbone_FFN import LatentMoE, SwiGLUExperts, TopKRouter, _new_float32_scalar
from .backbone_Mixer import (
    GDN2Block,
    GDN2MoEBlock,
    Mamba3Block,
    Mamba3MoEBlock,
    RavenBlock,
    RavenMoEBlock,
    SequenceFeedForwardBlock,
    StandardAttentionBlock,
    StandardAttentionMoEBlock,
)
from .configuration import AttentionBackboneConfig

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


@dataclass
class AttentionBackboneOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache | list[torch.Tensor]] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[object | None, ...]] = None
    aux_loss: Optional[torch.Tensor] = None


class AttentionBackbonePreTrainedModel(PreTrainedModel):
    config_class = AttentionBackboneConfig
    supports_gradient_checkpointing = True
    base_model_prefix = "backbone"
    _no_split_modules = [
        "RavenBlock",
        "RavenMoEBlock",
        "GDN2Block",
        "GDN2MoEBlock",
        "Mamba3Block",
        "Mamba3MoEBlock",
        "StandardAttentionBlock",
        "StandardAttentionMoEBlock",
    ]
    _supports_cache_class = True

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ) -> None:
        std = self.config.initializer_range
        if isinstance(module, GatedDeltaNet2) and next(module.parameters()).device.type != "meta":
            with torch.no_grad():
                if not getattr(module.A_log, "_is_hf_initialized", False):
                    module.A_log.copy_(nn.init.uniform_(module.A_log, a=0, b=16).log())
                module.A_log._no_weight_decay = True
                if not getattr(module.dt_bias, "_is_hf_initialized", False):
                    dt = torch.exp(
                        nn.init.uniform_(module.dt_bias)
                        * (math.log(0.1) - math.log(0.001))
                        + math.log(0.001),
                    ).clamp(min=1e-4)
                    inv_dt = dt + torch.log(-torch.expm1(-dt))
                    module.dt_bias.copy_(inv_dt)
                module.dt_bias._no_weight_decay = True
        elif isinstance(module, Mamba3) and next(module.parameters()).device.type != "meta":
            if not getattr(module.dt_bias, "_is_hf_initialized", False):
                dt = torch.exp(
                    torch.rand(module.num_heads)
                    * (math.log(module.dt_max) - math.log(module.dt_min))
                    + math.log(module.dt_min),
                ).clamp(min=module.dt_init_floor)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                with torch.no_grad():
                    module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True
            module.dt_bias._no_weight_decay = True

            if not getattr(module.D, "_is_hf_initialized", False):
                nn.init.ones_(module.D)
            module.D._no_weight_decay = True

            for parameter in (module.B_bias, module.C_bias):
                if not getattr(parameter, "_is_hf_initialized", False):
                    nn.init.ones_(parameter)

            if module.is_mimo:
                if not getattr(module.mimo_x, "_is_hf_initialized", False):
                    with torch.no_grad():
                        module.mimo_x.fill_(1.0 / module.mimo_rank)
                if not getattr(module.mimo_z, "_is_hf_initialized", False):
                    nn.init.ones_(module.mimo_z)
                if not getattr(module.mimo_o, "_is_hf_initialized", False):
                    with torch.no_grad():
                        module.mimo_o.fill_(1.0 / module.mimo_rank)

            if self.config.mamba3_config.rescale_prenorm_residual:
                # Mamba3 has one mixer residual branch per layer. Match FLA's
                # depth-aware initialization for its output projection.
                output_weight = module.out_proj.weight
                if not getattr(output_weight, "_is_hf_initialized", False):
                    nn.init.kaiming_uniform_(output_weight, a=math.sqrt(5))
                    with torch.no_grad():
                        output_weight /= math.sqrt(self.config.num_hidden_layers)
        elif isinstance(module, SwiGLUExperts):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)
            if module.gate_up_bias is not None:
                nn.init.zeros_(module.gate_up_bias)
                nn.init.zeros_(module.down_bias)
        elif isinstance(module, TopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            if getattr(module, "_is_attnres_proj", False):
                nn.init.zeros_(module.weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.RMSNorm) or module.__class__.__name__ == "RMSNorm":
            nn.init.ones_(module.weight)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            p = None
            if hasattr(module, "o_proj"):
                p = module.o_proj.weight
            elif hasattr(module, "out_proj"):
                p = module.out_proj.weight
            elif hasattr(module, "down_proj"):
                down_proj = module.down_proj
                if isinstance(down_proj, nn.Parameter) or torch.is_tensor(down_proj):
                    p = down_proj
                elif hasattr(down_proj, "weight"):
                    p = down_proj.weight
            if p is not None and not getattr(p, "_is_hf_initialized", False):
                if prenorm_residual_strategy == "rescale":
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == "zero":
                    nn.init.zeros_(p)
                else:
                    raise ValueError(
                        f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}"
                    )


class AttentionBackboneSummaryMixin:
    def parameter_name_summary(self) -> List[Dict[str, object]]:
        summary = []
        layer_prefix = "layers."
        for name, parameter in self.named_parameters():
            item = {
                "name": name,
                "shape": tuple(parameter.shape),
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "requires_grad": parameter.requires_grad,
            }
            if name.startswith(layer_prefix):
                parts = name[len(layer_prefix):].split(".", 2)
                if len(parts) >= 2 and parts[0].isdigit():
                    layer_idx = int(parts[0])
                    layer = self.layers[layer_idx]
                    item.update(
                        {
                            "layer_idx": layer_idx,
                            "layer_kind": self.layer_kinds[layer_idx],
                            "block": layer.__class__.__name__,
                            "attn": layer.attn.__class__.__name__,
                            "mlp": layer.mlp.__class__.__name__,
                            "parameter_group": parts[1],
                        }
                    )
            elif name.startswith("final_norm."):
                item.update(
                    {
                        "layer_idx": None,
                        "layer_kind": "final_norm",
                        "block": self.final_norm.__class__.__name__,
                        "attn": None,
                        "mlp": None,
                        "parameter_group": "final_norm",
                    }
                )
            summary.append(item)
        return summary

    def trainable_parameter_summary(self) -> List[Dict[str, object]]:
        summary = []
        for item in self.parameter_name_summary():
            parameter = dict(item)
            if not parameter.pop("requires_grad"):
                continue
            parameter["numel"] = self.get_parameter(parameter["name"]).numel()
            parameter["meaning"] = "attention backbone trainable parameter"
            summary.append(parameter)
        return summary

    def architecture_summary(self) -> str:
        lines = [
            "AttentionBackboneModel",
            (
                "  model: "
                f"AttentionBackboneModel(hidden_size={self.hidden_size}, "
                f"num_layers={len(self.layers)}, "
                f"architecture_type={self.architecture_type})"
            ),
            "    blocks:",
            *(
                f"      {spec_idx}: attn={spec['attn']}, mlp={spec['mlp']}, "
                f"num_layers={spec['num_layers']}"
                for spec_idx, spec in enumerate(self.block_specs)
            ),
            f"    return_aux_loss: {self.return_aux_loss}",
            f"    use_cache: {self.config.use_cache}",
            f"    recurrent_cycles: {self.num_recurrent_cycles}",
            f"    gradient_cycles: {self.num_gradient_cycles}",
            f"    post_norm_enabled: {self.use_post_norm}",
            f"    residual_dropout: {self.config.residual_dropout}",
            f"    attention_num_heads: {self.config.attention_config.num_heads}",
            f"    attention_num_kv_heads: {self.config.attention_config.num_kv_heads}",
            f"    standard_attention_heads: {self.attention_head_type or 'not used'}",
            "    residual_order: Norm -> attn-like mixer -> Add -> Norm -> MLP/MoE -> Add",
        ]
        for layer_idx, layer in enumerate(self.iter_layers()):
            layer_kind = self.layer_kinds[layer_idx]
            mixer = layer.attn
            mlp = layer.mlp
            lines.append(f"    layer[{layer_idx}]: {layer.__class__.__name__}({layer_kind})")
            lines.append(f"      attn_norm: {layer.attn_norm.__class__.__name__}")
            lines.append(f"      mlp_norm: {layer.mlp_norm.__class__.__name__}")
            lines.append(f"      use_cache: {layer.use_cache}")
            lines.append(f"      use_attnres: {layer.use_attnres}")
            attn_dropout_p = layer.attn_dropout.p if isinstance(layer.attn_dropout, nn.Dropout) else 0.0
            mlp_dropout_p = layer.mlp_dropout.p if isinstance(layer.mlp_dropout, nn.Dropout) else 0.0
            lines.append(
                f"      attn_dropout: {layer.attn_dropout.__class__.__name__}(p={attn_dropout_p})"
            )
            lines.append(
                f"      mlp_dropout: {layer.mlp_dropout.__class__.__name__}(p={mlp_dropout_p})"
            )
            lines.append(f"      gradient_checkpointing: {layer.gradient_checkpointing}")
            if isinstance(mixer, Raven):
                lines.append(
                    f"      fla: heads={mixer.num_heads}, "
                    f"head_dim={mixer.head_k_dim}, type=raven"
                )
                lines.append(
                    "      layer: "
                    f"{mixer.__class__.__name__}(mode={mixer.mode}, "
                    f"expand_k={mixer.expand_k}, expand_v={mixer.expand_v}, "
                    f"kv_heads={mixer.num_kv_heads}, topk={mixer.topk})"
                )
            elif isinstance(mixer, GatedDeltaNet2):
                lines.append(
                    f"      fla: heads={mixer.num_heads}, "
                    f"head_dim={mixer.head_dim}, type=gdn2"
                )
                lines.append(
                    "      layer: "
                    f"{mixer.__class__.__name__}(mode={mixer.mode}, "
                    f"expand_v={mixer.expand_v}, "
                    f"short_conv={mixer.use_short_conv})"
                )
            elif isinstance(mixer, Mamba3):
                mamba3_config = self.config.mamba3_config
                lines.append(
                    f"      fla: heads={mixer.num_heads}, "
                    f"head_dim={mamba3_config.head_dim}, type=mamba3"
                )
                lines.append(
                    "      layer: "
                    f"{mixer.__class__.__name__}(state_size={mamba3_config.state_size}, "
                    f"is_mimo={mamba3_config.is_mimo}, "
                    f"rescale_prenorm_residual={mamba3_config.rescale_prenorm_residual})"
                )
            elif isinstance(mixer, Attention):
                lines.append(
                    f"      attention: heads={mixer.num_heads}, "
                    f"kv_heads={mixer.num_kv_heads}, "
                    f"head_dim={mixer.head_dim}, type={self.attention_head_type}, "
                    f"window_size={self.config.attention_config.window_size}"
                )
            if isinstance(mlp, GatedMLP):
                lines.append(
                    f"      mlp: {mlp.__class__.__name__}"
                    f"(hidden={mlp.hidden_size}, hidden_ratio={mlp.hidden_ratio}, "
                    f"intermediate={mlp.intermediate_size}, hidden_act={mlp.hidden_act}, "
                    f"fuse_swiglu={mlp.fuse_swiglu})"
                )
            elif isinstance(mlp, LatentMoE):
                shared_intermediate = (
                    mlp.shared_expert.intermediate_size
                    if mlp.shared_expert is not None
                    else None
                )
                lines.append(
                    (
                        "      moe: "
                        f"experts={mlp.num_experts}, top_k={mlp.top_k}, "
                        f"hidden={mlp.hidden_size}, "
                        f"latent_hidden={mlp.latent_hidden_size}, "
                        f"latent_intermediate={mlp.latent_intermediate_size}, "
                        f"shared_intermediate={shared_intermediate}, "
                        f"mlp_hidden_ratio={mlp.mlp_hidden_ratio}, "
                        f"mlp_hidden_act={mlp.mlp_hidden_act}, "
                        f"mlp_fuse_swiglu={mlp.mlp_fuse_swiglu}, "
                        f"router={mlp.router.__class__.__name__ if mlp.router is not None else 'None'}, "
                        f"expert_impl={mlp.experts.__class__.__name__}, "
                        f"shared_expert={'yes' if mlp.shared_expert is not None else 'no'}"
                    )
                )
        lines.append(f"    final_norm: {self.final_norm.__class__.__name__}")
        return "\n".join(lines)

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

class AttentionBackboneModel(AttentionBackboneSummaryMixin, AttentionBackbonePreTrainedModel):
    def __init__(
        self,
        config: AttentionBackboneConfig,
        *,
        _hrm_module_role: str | None = None,
        _block_specs: list[dict[str, object]] | None = None,
    ) -> None:
        super().__init__(config)
        if _hrm_module_role not in {None, "H", "L"}:
            raise ValueError("_hrm_module_role must be None, 'H', or 'L'.")
        is_hrm_architecture = config.architecture_type in {"hrm", "heterogeneous_hrm"}
        if is_hrm_architecture and _hrm_module_role is None:
            raise NotImplementedError(
                "Construct an HRM architecture through build_backbone_model()."
            )
        if _hrm_module_role is not None and not is_hrm_architecture:
            raise ValueError("_hrm_module_role is only valid for an HRM architecture.")
        if _block_specs is not None and _hrm_module_role is None:
            raise ValueError("_block_specs overrides are only valid for an HRM module.")
        resolved_block_specs = _block_specs if _block_specs is not None else config.blocks
        if resolved_block_specs is None:
            raise ValueError("No block specifications were provided for the backbone module.")
        hidden_size = config.resolved_hidden_size
        self.architecture_type = (
            f"{config.architecture_type}_{_hrm_module_role.lower()}_module"
            if _hrm_module_role is not None
            else config.architecture_type
        )
        self.num_recurrent_cycles = (
            config.looped_num_cycles
            if self.architecture_type == "looped"
            else 1
        )
        self.num_gradient_cycles = (
            config.looped_num_gradient_cycles
            if self.architecture_type == "looped"
            else 1
        )
        self.use_post_norm = _hrm_module_role is not None or self.architecture_type == "looped"
        self.layers = nn.ModuleList()
        self.layer_kinds = []
        self.block_specs = [dict(spec) for spec in resolved_block_specs]
        self.num_hidden_layers = sum(int(spec["num_layers"]) for spec in self.block_specs)
        self.uses_standard_attention = any(spec["attn"] == "attention" for spec in self.block_specs)
        self.attention_head_type = (
            self._attention_head_type(config.attention_config.num_heads, config.attention_config.num_kv_heads)
            if self.uses_standard_attention
            else None
        )

        for spec in self.block_specs:
            for _ in range(int(spec["num_layers"])):
                self._append_block(config, str(spec["attn"]), str(spec["mlp"]))

        self.final_norm = (
            (RMSNorm if config.fuse_norm else nn.RMSNorm)(hidden_size, eps=config.norm_eps)
            if self.use_post_norm
            else nn.Identity()
        )

        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres and not self.use_post_norm:
            raise ValueError(
                "attnres_block_size requires a post-norm recurrent architecture."
            )
        if self.use_attnres:
            self.res_proj = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
            self.res_norm = nn.RMSNorm(normalized_shape=hidden_size, eps=config.norm_eps)
            self.res_proj._is_attnres_proj = True
        self.hidden_size = hidden_size
        self.return_aux_loss = config.return_aux_loss

        self.gradient_checkpointing = False

        self.post_init()

    @staticmethod
    def _attention_head_type(num_heads: int, num_kv_heads: Optional[int]) -> str:
        resolved_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert resolved_kv_heads > 0, "num_kv_heads must be positive."
        assert num_heads % resolved_kv_heads == 0, "num_heads must be divisible by num_kv_heads."
        if resolved_kv_heads == num_heads:
            return "mha"
        if resolved_kv_heads == 1:
            return "mqa"
        return "gqa"

    def _make_raven(self, config: AttentionBackboneConfig, layer_idx: int) -> Raven:
        raven_config = config.raven_config
        return Raven(
            hidden_size=config.resolved_hidden_size,
            num_heads=raven_config.num_heads,
            num_kv_heads=raven_config.num_kv_heads,
            num_slots=raven_config.num_slots,
            topk=raven_config.topk,
            norm_eps=config.norm_eps,
            feature_map=raven_config.feature_map,
            decay_type=raven_config.decay_type,
            router_score=raven_config.router_score,
            router_type=raven_config.router_type,
            layer_idx=layer_idx,
        )

    def _make_gdn2(self, config: AttentionBackboneConfig, layer_idx: int) -> GatedDeltaNet2:
        gdn2_config = config.gdn2_config
        return GatedDeltaNet2(
            hidden_size=config.resolved_hidden_size,
            num_heads=gdn2_config.num_heads,
            num_v_heads=gdn2_config.num_v_heads,
            head_dim=gdn2_config.head_dim,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
        )

    def _make_mamba3(self, config: AttentionBackboneConfig, layer_idx: int) -> Mamba3:
        if not hasattr(Mamba3, "_compute_a"):
            raise RuntimeError(
                "The installed flash-linear-attention Mamba3 has a broken A computation "
                "that makes the dd_A projection gradient identically zero. Upgrade FLA "
                "to an official revision that provides Mamba3._compute_a."
            )
        mamba3_config = config.mamba3_config
        return Mamba3(
            hidden_size=config.resolved_hidden_size,
            head_dim=mamba3_config.head_dim,
            state_size=mamba3_config.state_size,
            is_mimo=mamba3_config.is_mimo,
            use_bias=config.bias,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
        )

    def _make_attention(self, config: AttentionBackboneConfig, layer_idx: int) -> Attention:
        attention_config = config.attention_config
        return Attention(
            hidden_size=config.resolved_hidden_size,
            num_heads=attention_config.num_heads,
            num_kv_heads=attention_config.num_kv_heads,
            qkv_bias=config.bias,
            qk_norm=attention_config.qk_norm,
            window_size=attention_config.window_size,
            rope_theta=attention_config.rope_theta,
            max_position_embeddings=attention_config.max_position_embeddings,
            layer_idx=layer_idx,
        )

    def _make_attn(
        self,
        attn_type: str,
        config: AttentionBackboneConfig,
        layer_idx: int,
    ) -> Raven | GatedDeltaNet2 | Mamba3 | Attention:
        if attn_type == "raven":
            return self._make_raven(config, layer_idx)
        if attn_type == "gdn2":
            return self._make_gdn2(config, layer_idx)
        if attn_type == "mamba3":
            return self._make_mamba3(config, layer_idx)
        if attn_type == "attention":
            return self._make_attention(config, layer_idx)
        raise ValueError(f"Unsupported attn_type={attn_type!r}.")

    def _make_dense_mlp(self, config: AttentionBackboneConfig) -> GatedMLP:
        return GatedMLP(
            hidden_size=config.resolved_hidden_size,
            hidden_ratio=config.mlp_hidden_ratio,
            intermediate_size=config.mlp_intermediate_size,
            hidden_act=config.mlp_hidden_act,
            fuse_swiglu=config.mlp_fuse_swiglu,
        )

    def _make_moe(self, config: AttentionBackboneConfig) -> LatentMoE:
        return LatentMoE(
            hidden_size=config.resolved_hidden_size,
            latent_hidden_size=config.resolved_moe_latent_hidden_size,
            latent_intermediate_size=config.resolved_moe_latent_intermediate_size,
            shared_intermediate_size=config.mlp_intermediate_size,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            bias=config.bias,
            # Residual-branch dropout is applied once by SequenceFeedForwardBlock.
            dropout=0.0,
            capacity_factor=config.moe_capacity_factor,
            drop_tokens=config.moe_drop_tokens,
            moe_aux_loss_coeff=config.moe_aux_loss_coeff,
            use_seq_aux_loss=config.moe_use_seq_aux_loss,
            use_shared_expert=config.moe_use_shared_expert,
            return_aux_loss=config.return_aux_loss,
            mlp_hidden_ratio=config.mlp_hidden_ratio,
            mlp_hidden_act=config.mlp_hidden_act,
            mlp_fuse_swiglu=config.mlp_fuse_swiglu,
        )

    def _make_mlp(self, mlp_type: str, config: AttentionBackboneConfig) -> GatedMLP | LatentMoE:
        if mlp_type == "mlp":
            return self._make_dense_mlp(config)
        if mlp_type == "moe":
            return self._make_moe(config)
        raise ValueError(f"Unsupported mlp_type={mlp_type!r}.")

    @staticmethod
    def _block_class(attn_type: str, mlp_type: str) -> type[SequenceFeedForwardBlock]:
        block_classes = {
            ("raven", "mlp"): RavenBlock,
            ("raven", "moe"): RavenMoEBlock,
            ("gdn2", "mlp"): GDN2Block,
            ("gdn2", "moe"): GDN2MoEBlock,
            ("mamba3", "mlp"): Mamba3Block,
            ("mamba3", "moe"): Mamba3MoEBlock,
            ("attention", "mlp"): StandardAttentionBlock,
            ("attention", "moe"): StandardAttentionMoEBlock,
        }
        return block_classes[(attn_type, mlp_type)]

    def _append_block(
        self,
        config: AttentionBackboneConfig,
        attn_type: str,
        mlp_type: str,
    ) -> None:
        layer_idx = len(self.layers)
        block_cls = self._block_class(attn_type, mlp_type)
        self.layers.append(
            block_cls(
                config=config,
                layer_idx=layer_idx,
                attn=self._make_attn(attn_type, config, layer_idx),
                mlp=self._make_mlp(mlp_type, config),
            )
        )
        self.layer_kinds.append(f"{attn_type}_{mlp_type}")

    def iter_layers(self):
        return iter(self.layers)

    def iter_cache_layers(self):
        return iter(self.layers)

    def _apply_final_norm(
        self,
        hidden_states: torch.Tensor,
        attnres_states: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        if self.use_attnres:
            residuals = [*attnres_states, hidden_states]
            return fused_attnres(
                query=self.res_proj.weight,
                residuals=residuals,
                rms_weight=self.res_norm.weight,
                output_rms_weight=self.final_norm.weight,
                rms_eps=self.res_norm.eps,
            )
        return self.final_norm(hidden_states)

    def _forward_cycle(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | list[torch.Tensor] | None,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        all_hidden_states: tuple[torch.Tensor, ...] | None,
        all_attns: tuple[object | None, ...] | None,
        collect_aux_loss: bool,
    ) -> tuple[
        torch.Tensor,
        Cache | list[torch.Tensor] | None,
        tuple[torch.Tensor, ...] | None,
        tuple[object | None, ...] | None,
        list[torch.Tensor],
    ]:
        attnres_states = None
        cycle_aux_losses = []
        for layer in self.iter_layers():
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)
            # FLA mixers and FLA Attention share the same output slot, but current FLA
            # Attention does not materialize attention maps.
            layer_output_attentions = False
            outputs, layer_aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=layer_output_attentions,
                attnres_states=attnres_states,
            )
            hidden_states, attentions, past_key_values, attnres_states = outputs
            if output_attentions:
                all_attns = (*all_attns, attentions)
            if collect_aux_loss and layer_aux_loss is not None:
                cycle_aux_losses.append(layer_aux_loss)

        # A recurrent module call is the complete shared backbone followed by
        # its shared post norm. Baseline intentionally skips this post norm.
        if self.use_post_norm:
            hidden_states = self._apply_final_norm(hidden_states, attnres_states)

        return (
            hidden_states,
            past_key_values,
            all_hidden_states,
            all_attns,
            cycle_aux_losses,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache | list[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[dict],
    ) -> tuple | AttentionBackboneOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        hidden_states = input_ids
        if hidden_states.ndim != 3:
            raise ValueError("input_ids must have shape [batch_size, seq_len, hidden_size].")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"input_ids last dimension must match hidden_size={self.config.hidden_size}, "
                f"got {hidden_states.shape[-1]}."
            )

        if self.architecture_type == "looped" and (use_cache or past_key_values is not None):
            raise ValueError(
                "architecture_type='looped' does not support cache yet; "
                "set use_cache=False and past_key_values=None."
            )

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        aux_losses = []
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        # Looped architecture: reuse the same Backbone + PostNorm module and
        # therefore the same self.layers/final_norm parameters for all N cycles.
        outer_grad_enabled = torch.is_grad_enabled()
        first_gradient_cycle = self.num_recurrent_cycles - self.num_gradient_cycles
        for cycle_idx in range(self.num_recurrent_cycles):
            # The first N-K cycles run without autograd and only provide the recurrent state.
            # The final K cycles keep autograd enabled for truncated backpropagation.
            is_gradient_cycle = cycle_idx >= first_gradient_cycle
            cycle_grad_enabled = outer_grad_enabled and is_gradient_cycle
            with torch.set_grad_enabled(cycle_grad_enabled):
                (
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attns,
                    cycle_aux_losses,
                ) = self._forward_cycle(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    all_hidden_states=all_hidden_states,
                    all_attns=all_attns,
                    collect_aux_loss=cycle_grad_enabled,
                )
            aux_losses.extend(cycle_aux_losses)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if aux_losses:
            aux_loss = torch.stack([loss.to(torch.float32) for loss in aux_losses]).mean()
        else:
            aux_loss = None

        if not return_dict:
            return tuple(
                item
                for item in (
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attns,
                    aux_loss,
                )
                if item is not None
            )
        return AttentionBackboneOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            aux_loss=aux_loss,
        )


class HRMBackboneModel(AttentionBackbonePreTrainedModel):
    """H/L recurrent backbone with independent, internally shared modules."""

    backbone_architecture_type = "hrm"

    def __init__(self, config: AttentionBackboneConfig) -> None:
        super().__init__(config)
        if config.architecture_type != self.backbone_architecture_type:
            raise ValueError(
                f"{self.__class__.__name__} requires "
                f"architecture_type={self.backbone_architecture_type!r}."
            )

        if self.backbone_architecture_type == "heterogeneous_hrm":
            h_block_specs = config.hrm_h_blocks
            l_block_specs = config.hrm_l_blocks
        else:
            h_block_specs = config.blocks
            l_block_specs = config.blocks
        if h_block_specs is None or l_block_specs is None:
            raise ValueError("Both H and L block specifications must be available.")

        # H and L use the same architecture config but are distinct module instances.
        # Each instance owns a separate Backbone + PostNorm parameter set.
        self.H_level = AttentionBackboneModel(
            config,
            _hrm_module_role="H",
            _block_specs=h_block_specs,
        )
        self.L_level = AttentionBackboneModel(
            config,
            _hrm_module_role="L",
            _block_specs=l_block_specs,
        )

        self.hidden_size = config.resolved_hidden_size
        self.h_block_specs = [dict(spec) for spec in h_block_specs]
        self.l_block_specs = [dict(spec) for spec in l_block_specs]
        self.h_cycles = config.hrm_h_cycles
        self.l_cycles = config.hrm_l_cycles
        self.num_recurrent_cycles = self.h_cycles * (self.l_cycles + 1)
        self.num_gradient_cycles = config.hrm_num_gradient_steps
        self.use_post_norm = True
        self.return_aux_loss = config.return_aux_loss
        self.gradient_checkpointing = False

        self.register_buffer(
            "z_l_init",
            nn.init.trunc_normal_(torch.empty(self.hidden_size), std=1.0),
            persistent=True,
        )

    @property
    def module_sequence(self) -> str:
        cycle = "L" * self.l_cycles + "H"
        return " ".join(cycle for _ in range(self.h_cycles))

    def parameter_name_summary(self) -> List[Dict[str, object]]:
        summary = []
        for name, parameter in self.named_parameters():
            level = "H" if name.startswith("H_level.") else "L" if name.startswith("L_level.") else None
            summary.append(
                {
                    "name": name,
                    "shape": tuple(parameter.shape),
                    "dtype": str(parameter.dtype).replace("torch.", ""),
                    "requires_grad": parameter.requires_grad,
                    "level": level,
                }
            )
        return summary

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def architecture_summary(self) -> str:
        model_name = self.__class__.__name__
        lines = [
            model_name,
            (
                f"  model: {model_name}(hidden_size={self.hidden_size}, "
                f"H_layers={self.H_level.num_hidden_layers}, "
                f"L_layers={self.L_level.num_hidden_layers})"
            ),
            f"    architecture_type: {self.config.architecture_type}",
            f"    recurrence: H{self.h_cycles}L{self.l_cycles}",
            f"    module_sequence: {self.module_sequence}",
            "    state_flow: L evolves independently; H fuses previous-H + final-L; next group starts from H",
            f"    gradient_steps: last {self.num_gradient_cycles}/{self.num_recurrent_cycles}",
            "    input_gradient_bridge: backbone input -> first gradient-enabled module",
            "    parameter_sharing: H shared across H calls; L shared across L calls; H and L independent",
            f"    use_cache: {self.config.use_cache}",
            f"    H_level.post_norm: {self.H_level.final_norm.__class__.__name__}",
            f"    L_level.post_norm: {self.L_level.final_norm.__class__.__name__}",
            "    H_blocks:",
            *(
                f"      {spec_idx}: attn={spec['attn']}, mlp={spec['mlp']}, "
                f"num_layers={spec['num_layers']}"
                for spec_idx, spec in enumerate(self.h_block_specs)
            ),
            "    L_blocks:",
            *(
                f"      {spec_idx}: attn={spec['attn']}, mlp={spec['mlp']}, "
                f"num_layers={spec['num_layers']}"
                for spec_idx, spec in enumerate(self.l_block_specs)
            ),
        ]
        return "\n".join(lines)

    @staticmethod
    def _extend_optional_tuple(
        current: tuple | None,
        values: tuple | None,
    ) -> tuple | None:
        if current is None or values is None:
            return current
        return (*current, *values)

    @staticmethod
    def _bridge_input_gradient(
        module_input: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Keep the warm state value while routing truncated gradients to the input."""

        return module_input + (input_ids - input_ids.detach())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache | list[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[dict],
    ) -> tuple | AttentionBackboneOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if input_ids.ndim != 3:
            raise ValueError("input_ids must have shape [batch_size, seq_len, hidden_size].")
        if input_ids.shape[-1] != self.hidden_size:
            raise ValueError(
                f"input_ids last dimension must match hidden_size={self.hidden_size}, "
                f"got {input_ids.shape[-1]}."
            )
        if use_cache or past_key_values is not None:
            raise ValueError(
                f"architecture_type={self.config.architecture_type!r} does not support cache yet; "
                "set use_cache=False and past_key_values=None."
            )

        z_H = input_ids
        z_L = self.z_l_init.to(device=input_ids.device, dtype=input_ids.dtype)
        z_L = z_L.view(1, 1, -1).expand_as(input_ids)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        aux_losses = []
        outer_grad_enabled = torch.is_grad_enabled()
        first_gradient_step = self.num_recurrent_cycles - self.num_gradient_cycles
        step_idx = 0

        # Complete module calls before first_gradient_step run without autograd;
        # the final K L/H calls retain one contiguous truncated graph.
        for _h_idx in range(self.h_cycles):
            for _l_idx in range(self.l_cycles):
                # L evolves its own recurrent state while z_H remains the
                # shared high-level state for the current H cycle.
                step_grad_enabled = outer_grad_enabled and step_idx >= first_gradient_step
                with torch.set_grad_enabled(step_grad_enabled):
                    module_input = z_L
                    if first_gradient_step > 0 and step_idx == first_gradient_step:
                        # Warm-up modules stay outside autograd. This zero-valued
                        # straight-through term reconnects the final K-module
                        # graph to the trainable upstream frontend without
                        # changing the recurrent state's forward value.
                        module_input = self._bridge_input_gradient(
                            module_input,
                            input_ids,
                        )
                    level_outputs = self.L_level(
                        input_ids=module_input,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=True,
                    )
                    z_L = level_outputs.last_hidden_state
                all_hidden_states = self._extend_optional_tuple(
                    all_hidden_states,
                    level_outputs.hidden_states,
                )
                all_attns = self._extend_optional_tuple(all_attns, level_outputs.attentions)
                if level_outputs.aux_loss is not None:
                    aux_losses.append(level_outputs.aux_loss)
                step_idx += 1

            # Fuse the preserved high-level state with the final L state. H is
            # distinct from L, but each H call reuses the same parameter set.
            step_grad_enabled = outer_grad_enabled and step_idx >= first_gradient_step
            with torch.set_grad_enabled(step_grad_enabled):
                module_input = z_H + z_L
                if first_gradient_step > 0 and step_idx == first_gradient_step:
                    module_input = self._bridge_input_gradient(
                        module_input,
                        input_ids,
                    )
                level_outputs = self.H_level(
                    input_ids=module_input,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
                z_H = level_outputs.last_hidden_state
            all_hidden_states = self._extend_optional_tuple(
                all_hidden_states,
                level_outputs.hidden_states,
            )
            all_attns = self._extend_optional_tuple(all_attns, level_outputs.attentions)
            if level_outputs.aux_loss is not None:
                aux_losses.append(level_outputs.aux_loss)
            step_idx += 1

            # The completed H state is the common starting state for both
            # branches in the next group: L starts from it, while H preserves it
            # until the next final-L fusion.
            if _h_idx + 1 < self.h_cycles:
                z_L = z_H

        aux_loss = (
            torch.stack([loss.to(torch.float32) for loss in aux_losses]).mean()
            if aux_losses
            else None
        )

        if not return_dict:
            return tuple(
                item
                for item in (
                    z_H,
                    all_hidden_states,
                    all_attns,
                    aux_loss,
                )
                if item is not None
            )
        return AttentionBackboneOutput(
            last_hidden_state=z_H,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            aux_loss=aux_loss,
        )


class HeterogeneousHRMBackboneModel(HRMBackboneModel):
    """HRM backbone with independently configured H and L module topologies."""

    backbone_architecture_type = "heterogeneous_hrm"


def build_backbone_model(
    config: AttentionBackboneConfig,
) -> AttentionBackboneModel | HRMBackboneModel | HeterogeneousHRMBackboneModel:
    if config.architecture_type == "hrm":
        return HRMBackboneModel(config)
    if config.architecture_type == "heterogeneous_hrm":
        return HeterogeneousHRMBackboneModel(config)
    return AttentionBackboneModel(config)
