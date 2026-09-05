from __future__ import annotations

import torch
from fla.models.utils import Cache

from ....configuration.components.backbone import AttentionBackboneConfig
from ..backbone_stack import BackboneStack
from .base import ArchitectureOutput, BackboneArchitecture, mean_aux_loss


class BaselineArchitecture(BackboneArchitecture):
    """One non-recurrent pre-norm backbone. No terminal post-norm."""

    supports_cache = True

    def __init__(self, config: AttentionBackboneConfig) -> None:
        super().__init__()
        self.stack = BackboneStack(
            config,
            config.block_specs,
            use_post_norm=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | list[torch.Tensor] | None,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> ArchitectureOutput:
        output = self.stack(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            collect_aux_loss=torch.is_grad_enabled(),
        )
        return ArchitectureOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            aux_loss=mean_aux_loss(list(output.aux_losses)),
        )
