from __future__ import annotations

import torch
from torch import nn

from ...configuration.components.pooling import GatedAttentionPoolingConfig
from .base import PoolingOutputTransform, validate_pooling_input


class GlobalAvgMaxPooling(nn.Module):
    """Project concatenated global mean and maximum statistics."""

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__()
        self.input_size = config.input_size
        self.projection = nn.Linear(
            2 * config.input_size,
            config.output_size,
            bias=config.bias,
        )
        self.output_transform = PoolingOutputTransform(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        validate_pooling_input(hidden_states, self.input_size)
        statistics = torch.cat(
            (hidden_states.mean(dim=1), hidden_states.amax(dim=1)),
            dim=-1,
        )
        return self.output_transform(self.projection(statistics))
