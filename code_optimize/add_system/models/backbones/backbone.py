from __future__ import annotations

import math
from typing import ClassVar

import torch
from fla.layers import GatedDeltaNet2, Mamba3
from fla.models.utils import Cache
from torch import nn
from transformers import PreTrainedModel

from ...configuration.components.backbone import AttentionBackboneConfig
from ..outputs import BackboneOutput
from .architectures.factory import build_architecture
from .feedforward.experts import SwiGLUExperts
from .feedforward.router import TopKRouter


class BackboneModel(PreTrainedModel):
    """HF wrapper around one explicitly selected backbone architecture."""

    config_class = AttentionBackboneConfig
    supports_gradient_checkpointing = True
    base_model_prefix = "backbone"
    _supports_cache_class = True
    _no_split_modules: ClassVar[list[str]] = ["SequenceFeedForwardBlock"]

    def __init__(self, config: AttentionBackboneConfig) -> None:
        super().__init__(config)
        self.architecture = build_architecture(config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, GatedDeltaNet2):
            if next(module.parameters()).device.type == "meta":
                return
            with torch.no_grad():
                if not getattr(module.A_log, "_is_hf_initialized", False):
                    module.A_log.copy_(nn.init.uniform_(module.A_log, 0, 16).log())
                if not getattr(module.dt_bias, "_is_hf_initialized", False):
                    dt = torch.exp(
                        nn.init.uniform_(module.dt_bias) * (math.log(0.1) - math.log(0.001))
                        + math.log(0.001)
                    ).clamp_min(1e-4)
                    module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            module.A_log._no_weight_decay = True
            module.dt_bias._no_weight_decay = True
        elif isinstance(module, Mamba3):
            if next(module.parameters()).device.type == "meta":
                return
            with torch.no_grad():
                if not getattr(module.dt_bias, "_is_hf_initialized", False):
                    dt = torch.exp(
                        torch.rand_like(module.dt_bias)
                        * (math.log(module.dt_max) - math.log(module.dt_min))
                        + math.log(module.dt_min)
                    ).clamp_min(module.dt_init_floor)
                    module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
                if not getattr(module.D, "_is_hf_initialized", False):
                    module.D.fill_(1.0)
                for parameter in (module.B_bias, module.C_bias):
                    if not getattr(parameter, "_is_hf_initialized", False):
                        parameter.fill_(1.0)
                if module.is_mimo:
                    module.mimo_x.fill_(1.0 / module.mimo_rank)
                    module.mimo_z.fill_(1.0)
                    module.mimo_o.fill_(1.0 / module.mimo_rank)
            module.dt_bias._no_reinit = True
            module.dt_bias._no_weight_decay = True
            module.D._no_weight_decay = True
            if self.config.mamba3_config.rescale_prenorm_residual:
                weight = module.out_proj.weight
                if not getattr(weight, "_is_hf_initialized", False):
                    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                    with torch.no_grad():
                        weight /= math.sqrt(self.config.num_hidden_layers)
        elif isinstance(module, SwiGLUExperts):
            nn.init.normal_(module.gate_up_projection, mean=0.0, std=std)
            nn.init.normal_(module.down_projection, mean=0.0, std=std)
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
        elif isinstance(module, nn.RMSNorm) or module.__class__.__name__ == "RMSNorm":
            nn.init.ones_(module.weight)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.Tensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BackboneOutput:
        if input_ids.ndim != 3 or input_ids.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"input_ids must have shape [batch, sequence, {self.config.hidden_size}], "
                f"got {tuple(input_ids.shape)}."
            )
        resolved_use_cache = self.config.use_cache if use_cache is None else use_cache
        resolved_output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions
        )
        resolved_output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        resolved_return_dict = self.config.return_dict if return_dict is None else return_dict
        if resolved_use_cache and not self.architecture.supports_cache:
            raise ValueError(
                f"{self.config.architecture_type} architecture does not support cache."
            )
        if resolved_use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        output = self.architecture(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=resolved_use_cache,
            output_attentions=resolved_output_attentions,
            output_hidden_states=resolved_output_hidden_states,
        )
        if not resolved_return_dict:
            return tuple(
                item
                for item in (
                    output.last_hidden_state,
                    output.past_key_values,
                    output.hidden_states,
                    output.attentions,
                    output.aux_loss,
                )
                if item is not None
            )
        return BackboneOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            aux_loss=output.aux_loss,
        )
