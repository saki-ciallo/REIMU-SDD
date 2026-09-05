from __future__ import annotations

import torch
from torch import nn

from ...configuration.components.pooling import GatedAttentionPoolingConfig
from .base import PoolingOutputTransform, validate_pooling_input


class GatedAttentionPoolingV1(nn.Module):
    """SwiGLU temporal scorer that pools the original sequence values."""

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__()
        self.input_size = config.input_size
        hidden_size = max(1, round(config.hidden_ratio * config.input_size))
        self.score_temperature = config.score_temperature
        self.gate_value_projection = nn.Linear(
            config.input_size,
            2 * hidden_size,
            bias=False,
        )
        self.score_projection = nn.Linear(hidden_size, 1, bias=False)
        self.output_projection = (
            nn.Linear(config.input_size, config.output_size, bias=config.bias)
            if config.input_size != config.output_size
            else nn.Identity()
        )
        self.activation = nn.SiLU()
        self.output_transform = PoolingOutputTransform(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        validate_pooling_input(hidden_states, self.input_size)
        gates, values = self.gate_value_projection(hidden_states).chunk(2, dim=-1)
        logits = self.score_projection(self.activation(gates) * values).squeeze(-1)
        weights = torch.softmax(
            logits.float() / self.score_temperature,
            dim=1,
        ).to(hidden_states.dtype)
        pooled = torch.bmm(weights[:, None, :], hidden_states).squeeze(1)
        return self.output_transform(self.output_projection(pooled))
