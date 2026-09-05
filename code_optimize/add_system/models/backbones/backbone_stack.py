from __future__ import annotations

from dataclasses import dataclass

import torch
from fla.models.utils import Cache
from fla.modules import RMSNorm
from fla.ops.attnres import fused_attnres
from torch import nn

from ...configuration.blocks import BlockSpec, expand_block_specs
from ...configuration.components.backbone import AttentionBackboneConfig
from .blocks import SequenceFeedForwardBlock

type LayerCache = Cache | list[torch.Tensor] | None


@dataclass(slots=True)
class StackOutput:
    last_hidden_state: torch.Tensor
    past_key_values: LayerCache
    hidden_states: tuple[torch.Tensor, ...] | None
    attentions: tuple[object | None, ...] | None
    aux_losses: tuple[torch.Tensor, ...]


class BackboneStack(nn.Module):
    """One complete shared backbone module, with optional terminal post-norm."""

    def __init__(
        self,
        config: AttentionBackboneConfig,
        specifications: tuple[BlockSpec, ...],
        *,
        use_post_norm: bool,
    ) -> None:
        super().__init__()
        expanded = expand_block_specs(specifications)
        self.layers = nn.ModuleList(
            SequenceFeedForwardBlock(config, specification, index)
            for index, specification in enumerate(expanded)
        )
        norm_type = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.post_norm = (
            norm_type(config.hidden_size, eps=config.norm_eps) if use_post_norm else nn.Identity()
        )
        self.use_post_norm = use_post_norm
        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres and not use_post_norm:
            raise ValueError("attnres requires a recurrent architecture post-norm.")
        if self.use_attnres:
            self.residual_projection = nn.Linear(config.hidden_size, 1, bias=False)
            self.residual_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.residual_projection._is_attnres_proj = True

    def _apply_post_norm(
        self,
        hidden_states: torch.Tensor,
        attnres_states: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self.use_post_norm:
            return hidden_states
        if not self.use_attnres:
            return self.post_norm(hidden_states)
        residuals = [*(attnres_states or ()), hidden_states]
        return fused_attnres(
            query=self.residual_projection.weight,
            residuals=residuals,
            rms_weight=self.residual_norm.weight,
            output_rms_weight=self.post_norm.weight,
            rms_eps=self.residual_norm.eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        past_key_values: LayerCache,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        collect_aux_loss: bool,
    ) -> StackOutput:
        collected_hidden: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        collected_attentions: tuple[object | None, ...] | None = () if output_attentions else None
        aux_losses: list[torch.Tensor] = []
        attnres_states = None

        for layer in self.layers:
            if collected_hidden is not None:
                collected_hidden = (*collected_hidden, hidden_states)
            # FLA 0.5.2 exposes output_attentions but does not materialize a
            # tensor when it is true. Keep the call disabled and preserve one
            # None slot per layer for a stable mixed-mixer output contract.
            layer_output, layer_aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=False,
                attnres_states=attnres_states,
            )
            hidden_states, attention, past_key_values, attnres_states = layer_output
            if collected_attentions is not None:
                collected_attentions = (*collected_attentions, attention)
            if collect_aux_loss and layer_aux_loss is not None:
                aux_losses.append(layer_aux_loss)

        hidden_states = self._apply_post_norm(hidden_states, attnres_states)
        if collected_hidden is not None:
            collected_hidden = (*collected_hidden, hidden_states)
        return StackOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=collected_hidden,
            attentions=collected_attentions,
            aux_losses=tuple(aux_losses),
        )
