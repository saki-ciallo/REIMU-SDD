from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from fla.models.utils import Cache
from fla.modules import RMSNorm
from fla.ops.attnres import fused_attnres
from transformers.modeling_layers import GradientCheckpointingLayer

from .backbone_FFN import LatentMoE, MixerOutput
from .configuration import AttentionBackboneConfig

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class SequenceFeedForwardBlock(GradientCheckpointingLayer):
    """
    FLA-style block: norm -> attn-like mixer -> add -> norm -> MLP/MoE -> add.
    可通过外部传入 attention 和 mlp 的类型来构建不同的 block
    """

    attn_name = "attention"
    mlp_name = "mlp"

    def __init__(
        self,
        config: AttentionBackboneConfig,
        layer_idx: int,
        attn: nn.Module,
        mlp: nn.Module,
    ) -> None:
        super().__init__()
        hidden_size = config.resolved_hidden_size
        norm_cls = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.config = config
        self.layer_idx = layer_idx
        self.use_cache = config.use_cache
        self.attn_norm = norm_cls(hidden_size, eps=config.norm_eps)
        self.attn = attn
        self.mlp_norm = norm_cls(hidden_size, eps=config.norm_eps)
        self.mlp = mlp
        self.hidden_size = hidden_size
        self.attn_dropout = (
            nn.Dropout(config.residual_dropout)
            if config.residual_dropout > 0.0
            else nn.Identity()
        )
        self.mlp_dropout = (
            nn.Dropout(config.residual_dropout)
            if config.residual_dropout > 0.0
            else nn.Identity()
        )

        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            self.attn_res_proj = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
            self.attn_res_norm = nn.RMSNorm(normalized_shape=hidden_size, eps=config.norm_eps)
            self.mlp_res_proj = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
            self.mlp_res_norm = nn.RMSNorm(normalized_shape=hidden_size, eps=config.norm_eps)
            block_size = config.attnres_block_size
            self.attnres_is_attn_boundary = (2 * layer_idx) % block_size == 0
            self.attnres_is_mlp_boundary = (2 * layer_idx + 1) % block_size == 0
            self.attn_res_proj._is_attnres_proj = True
            self.mlp_res_proj._is_attnres_proj = True

    def _forward_mlp(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = self.mlp(hidden_states)
        if isinstance(output, MixerOutput):
            layer_aux_loss = (
                output.aux_loss
                if (
                    output.aux_loss is not None
                    and isinstance(self.mlp, LatentMoE)
                    and self.mlp.training
                    and self.mlp.return_aux_loss
                    and self.mlp.moe_aux_loss_coeff != 0.0
                )
                else None
            )
            return output.hidden_states, layer_aux_loss
        return output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        attnres_states: list[torch.Tensor] | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[
        tuple[
            torch.FloatTensor,
            object | None,
            Cache | list[torch.FloatTensor] | None,
            list[torch.Tensor] | None,
        ],
        torch.Tensor | None,
    ]:
        if hidden_states.ndim != 3:
            raise ValueError("backbone block expects hidden_states with shape [batch, seq, hidden].")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"backbone block hidden size mismatch: expected {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}."
            )

        if self.use_attnres:
            prefix_sum = hidden_states
            if attnres_states is None:
                hidden_states = self.attn_norm(prefix_sum)
                attnres_states = [prefix_sum]
                prefix_sum = None
            else:
                residuals = [*attnres_states, prefix_sum]
                if self.attnres_is_attn_boundary:
                    attnres_states = residuals
                    prefix_sum = None
                hidden_states = fused_attnres(
                    query=self.attn_res_proj.weight,
                    residuals=residuals,
                    rms_weight=self.attn_res_norm.weight,
                    output_rms_weight=self.attn_norm.weight,
                    rms_eps=self.attn_res_norm.eps,
                )
        else:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = self.attn_dropout(hidden_states)

        if self.use_attnres:
            prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
            residuals = [*attnres_states, prefix_sum]
            if self.attnres_is_mlp_boundary:
                attnres_states = residuals
                prefix_sum = None
            hidden_states = fused_attnres(
                query=self.mlp_res_proj.weight,
                residuals=residuals,
                rms_weight=self.mlp_res_norm.weight,
                output_rms_weight=self.mlp_norm.weight,
                rms_eps=self.mlp_res_norm.eps,
            )
        elif self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states, layer_aux_loss = self._forward_mlp(hidden_states) # 如果是 MoE 且使用 aux loss，这里会返回结果
        hidden_states = self.mlp_dropout(hidden_states)

        if self.use_attnres:
            hidden_states = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        else:
            hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values, attnres_states)
        return outputs, layer_aux_loss


class RavenBlock(SequenceFeedForwardBlock):
    attn_name = "raven"
    mlp_name = "mlp"


class RavenMoEBlock(SequenceFeedForwardBlock):
    attn_name = "raven"
    mlp_name = "moe"


class GDN2Block(SequenceFeedForwardBlock):
    attn_name = "gdn2"
    mlp_name = "mlp"


class GDN2MoEBlock(SequenceFeedForwardBlock):
    attn_name = "gdn2"
    mlp_name = "moe"


class Mamba3Block(SequenceFeedForwardBlock):
    attn_name = "mamba3"
    mlp_name = "mlp"


class Mamba3MoEBlock(SequenceFeedForwardBlock):
    attn_name = "mamba3"
    mlp_name = "moe"


class StandardAttentionBlock(SequenceFeedForwardBlock):
    attn_name = "attention"
    mlp_name = "mlp"


class StandardAttentionMoEBlock(SequenceFeedForwardBlock):
    attn_name = "attention"
    mlp_name = "moe"


__all__ = [
    "GDN2Block",
    "GDN2MoEBlock",
    "Mamba3Block",
    "Mamba3MoEBlock",
    "RavenBlock",
    "RavenMoEBlock",
    "SequenceFeedForwardBlock",
    "StandardAttentionBlock",
    "StandardAttentionMoEBlock",
]
