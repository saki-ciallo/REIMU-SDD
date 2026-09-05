from __future__ import annotations

import torch
from fla.models.utils import Cache
from fla.modules import RMSNorm
from fla.ops.attnres import fused_attnres
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

from ....configuration.blocks import BlockSpec
from ....configuration.components.backbone import AttentionBackboneConfig
from ..feedforward import LatentMoE, MoEOutput, build_feed_forward
from ..mixers import build_mixer

type LayerCache = Cache | list[torch.Tensor] | None
type AttnResStates = list[torch.Tensor] | None
type LayerOutput = tuple[
    torch.Tensor,
    object | None,
    LayerCache,
    AttnResStates,
]


class SequenceFeedForwardBlock(GradientCheckpointingLayer):
    """Pre-norm FLA mixer and MLP/MoE residual block."""

    def __init__(
        self,
        config: AttentionBackboneConfig,
        specification: BlockSpec,
        layer_index: int,
    ) -> None:
        super().__init__()
        norm_type = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.config = config
        self.specification = specification
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.attn_norm = norm_type(config.hidden_size, eps=config.norm_eps)
        self.attn = build_mixer(specification.mixer, config, layer_index)
        self.mlp_norm = norm_type(config.hidden_size, eps=config.norm_eps)
        self.mlp = build_feed_forward(specification.feed_forward, config)
        self.attn_dropout = nn.Dropout(config.residual_dropout)
        self.mlp_dropout = nn.Dropout(config.residual_dropout)

        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            block_size = config.attnres_block_size
            self.attn_res_projection = nn.Linear(config.hidden_size, 1, bias=False)
            self.attn_res_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.mlp_res_projection = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.attn_boundary = (2 * layer_index) % block_size == 0
            self.mlp_boundary = (2 * layer_index + 1) % block_size == 0
            self.attn_res_projection._is_attnres_proj = True
            self.mlp_res_projection._is_attnres_proj = True

    def _forward_feed_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = self.mlp(hidden_states)
        if not isinstance(output, MoEOutput):
            return output, None
        collect_aux = (
            isinstance(self.mlp, LatentMoE)
            and self.mlp.training
            and self.mlp.return_aux_loss
            and self.mlp.aux_loss_coefficient != 0.0
        )
        return output.hidden_states, output.aux_loss if collect_aux else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: LayerCache = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        attnres_states: AttnResStates = None,
    ) -> tuple[LayerOutput, torch.Tensor | None]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [batch, sequence, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}."
            )

        if self.use_attnres:
            prefix_sum = hidden_states
            if attnres_states is None:
                hidden_states = self.attn_norm(prefix_sum)
                attnres_states = [prefix_sum]
                prefix_sum = None
            else:
                residuals = [*attnres_states, prefix_sum]
                if self.attn_boundary:
                    attnres_states = residuals
                    prefix_sum = None
                hidden_states = fused_attnres(
                    query=self.attn_res_projection.weight,
                    residuals=residuals,
                    rms_weight=self.attn_res_norm.weight,
                    output_rms_weight=self.attn_norm.weight,
                    rms_eps=self.attn_res_norm.eps,
                )
        else:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)

        hidden_states, attention, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self.attn_dropout(hidden_states)

        if self.use_attnres:
            prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
            residuals = [*attnres_states, prefix_sum]
            if self.mlp_boundary:
                attnres_states = residuals
                prefix_sum = None
            hidden_states = fused_attnres(
                query=self.mlp_res_projection.weight,
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

        hidden_states, aux_loss = self._forward_feed_forward(hidden_states)
        hidden_states = self.mlp_dropout(hidden_states)
        hidden_states = (
            hidden_states
            if self.use_attnres and prefix_sum is None
            else prefix_sum + hidden_states
            if self.use_attnres
            else residual + hidden_states
        )
        return (
            hidden_states,
            attention,
            past_key_values,
            attnres_states,
        ), aux_loss
