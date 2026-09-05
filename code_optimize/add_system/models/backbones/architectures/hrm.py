from __future__ import annotations

import torch
from torch import nn

from ....configuration.blocks import BlockSpec
from ....configuration.components.backbone import AttentionBackboneConfig
from ..backbone_stack import BackboneStack
from .base import (
    ArchitectureOutput,
    BackboneArchitecture,
    bridge_input_gradient,
    extend_optional_tuple,
    mean_aux_loss,
)


class HRMArchitecture(BackboneArchitecture):
    """Distinct H/L modules with shared parameters across their own recurrent calls."""

    def __init__(
        self,
        config: AttentionBackboneConfig,
        *,
        h_specs: tuple[BlockSpec, ...] | None = None,
        l_specs: tuple[BlockSpec, ...] | None = None,
    ) -> None:
        super().__init__()
        resolved_h_specs = h_specs or config.block_specs
        resolved_l_specs = l_specs or config.block_specs
        self.high_level = BackboneStack(
            config,
            resolved_h_specs,
            use_post_norm=True,
        )
        self.low_level = BackboneStack(
            config,
            resolved_l_specs,
            use_post_norm=True,
        )
        self.hidden_size = config.hidden_size
        self.h_cycles = config.hrm_h_cycles
        self.l_cycles = config.hrm_l_cycles
        self.gradient_schedule = config.hrm_gradient_schedule
        self.register_buffer(
            "low_level_initial_state",
            nn.init.trunc_normal_(torch.empty(config.hidden_size), std=1.0),
        )

    def _run_level(
        self,
        stack: BackboneStack,
        module_input: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        step_index: int,
        first_gradient_step: int,
        step_gradient: bool,
        attention_mask: torch.Tensor | None,
        output_attentions: bool,
        output_hidden_states: bool,
    ):
        with torch.set_grad_enabled(step_gradient):
            if step_index == first_gradient_step and first_gradient_step > 0:
                module_input = bridge_input_gradient(module_input, input_ids)
            return stack(
                module_input,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                collect_aux_loss=step_gradient,
            )

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
            raise ValueError("HRM architectures do not support cache.")

        high_state = input_ids
        low_state = self.low_level_initial_state.to(
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        low_state = low_state.view(1, 1, -1).expand_as(input_ids)
        all_hidden: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        all_attentions: tuple[object | None, ...] | None = () if output_attentions else None
        aux_losses: list[torch.Tensor] = []
        outer_grad_enabled = torch.is_grad_enabled()
        first_gradient_step = self.gradient_schedule.index(True)
        step_index = 0

        # Per group: L evolves L times, then H consumes previous-H + final-L.
        for high_cycle in range(self.h_cycles):
            for _ in range(self.l_cycles):
                step_gradient = outer_grad_enabled and self.gradient_schedule[step_index]
                output = self._run_level(
                    self.low_level,
                    low_state,
                    input_ids=input_ids,
                    step_index=step_index,
                    first_gradient_step=first_gradient_step,
                    step_gradient=step_gradient,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                low_state = output.last_hidden_state
                all_hidden = extend_optional_tuple(all_hidden, output.hidden_states)
                all_attentions = extend_optional_tuple(
                    all_attentions,
                    output.attentions,
                )
                aux_losses.extend(output.aux_losses)
                step_index += 1

            step_gradient = outer_grad_enabled and self.gradient_schedule[step_index]
            output = self._run_level(
                self.high_level,
                high_state + low_state,
                input_ids=input_ids,
                step_index=step_index,
                first_gradient_step=first_gradient_step,
                step_gradient=step_gradient,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            high_state = output.last_hidden_state
            all_hidden = extend_optional_tuple(all_hidden, output.hidden_states)
            all_attentions = extend_optional_tuple(
                all_attentions,
                output.attentions,
            )
            aux_losses.extend(output.aux_losses)
            step_index += 1

            if high_cycle + 1 < self.h_cycles:
                low_state = high_state

        return ArchitectureOutput(
            last_hidden_state=high_state,
            hidden_states=all_hidden,
            attentions=all_attentions,
            aux_loss=mean_aux_loss(aux_losses),
        )
