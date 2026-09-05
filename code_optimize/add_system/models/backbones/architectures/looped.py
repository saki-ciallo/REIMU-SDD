from __future__ import annotations

import torch

from ....configuration.components.backbone import AttentionBackboneConfig
from ..backbone_stack import BackboneStack
from .base import (
    ArchitectureOutput,
    BackboneArchitecture,
    bridge_input_gradient,
    extend_optional_tuple,
    mean_aux_loss,
)


class LoopedArchitecture(BackboneArchitecture):
    """Reuse one complete ``BackboneStack + PostNorm`` module for N cycles."""

    def __init__(self, config: AttentionBackboneConfig) -> None:
        super().__init__()
        self.stack = BackboneStack(
            config,
            config.block_specs,
            use_post_norm=True,
        )
        self.gradient_schedule = config.looped_gradient_schedule

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        past_key_values: object | None,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> ArchitectureOutput:
        if use_cache or past_key_values is not None:
            raise ValueError("looped architecture does not support cache.")
        hidden_states = input_ids
        all_hidden: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        all_attentions: tuple[object | None, ...] | None = () if output_attentions else None
        aux_losses: list[torch.Tensor] = []
        outer_grad_enabled = torch.is_grad_enabled()
        first_gradient_cycle = self.gradient_schedule.index(True)

        for cycle_index, scheduled_gradient in enumerate(self.gradient_schedule):
            cycle_gradient = outer_grad_enabled and scheduled_gradient
            with torch.set_grad_enabled(cycle_gradient):
                module_input = hidden_states
                if cycle_index == first_gradient_cycle and first_gradient_cycle > 0:
                    module_input = bridge_input_gradient(module_input, input_ids)
                output = self.stack(
                    module_input,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    collect_aux_loss=cycle_gradient,
                )
                hidden_states = output.last_hidden_state
            all_hidden = extend_optional_tuple(all_hidden, output.hidden_states)
            all_attentions = extend_optional_tuple(
                all_attentions,
                output.attentions,
            )
            aux_losses.extend(output.aux_losses)

        return ArchitectureOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden,
            attentions=all_attentions,
            aux_loss=mean_aux_loss(aux_losses),
        )
