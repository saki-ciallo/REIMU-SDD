from __future__ import annotations

import torch
from torch import nn

from ...configuration.components.pooling import GatedAttentionPoolingConfig
from .base import PoolingOutputTransform, validate_pooling_input


class GatedAttentionPoolingV2(nn.Module):
    """SwiGLU temporal scorer with a separately projected pooling value."""

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__()
        self.input_size = config.input_size
        hidden_size = max(1, round(config.hidden_ratio * config.input_size))
        self.score_temperature = config.score_temperature
        self.projection = nn.Linear(config.input_size, 3 * hidden_size, bias=False)
        self.score_projection = nn.Linear(hidden_size, 1, bias=False)
        self.output_projection = nn.Linear(
            hidden_size,
            config.output_size,
            bias=config.bias,
        )
        self.activation = nn.SiLU()
        self.output_transform = PoolingOutputTransform(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        validate_pooling_input(hidden_states, self.input_size)
        gates, score_values, pool_values = self.projection(hidden_states).chunk(
            3,
            dim=-1,
        )
        logits = self.score_projection(self.activation(gates) * score_values).squeeze(-1)
        weights = torch.softmax(
            logits.float() / self.score_temperature,
            dim=1,
        ).to(pool_values.dtype)
        pooled = torch.bmm(weights[:, None, :], pool_values).squeeze(1)
        return self.output_transform(self.output_projection(pooled))
