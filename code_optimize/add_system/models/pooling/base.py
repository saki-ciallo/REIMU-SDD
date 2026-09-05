from __future__ import annotations

import torch
from fla.modules import RMSNorm
from torch import nn

from ...configuration.components.pooling import GatedAttentionPoolingConfig


def validate_pooling_input(hidden_states: torch.Tensor, input_size: int) -> None:
    expected = ("batch", "sequence", input_size)
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != input_size:
        raise ValueError(
            f"hidden_states must have shape [{expected[0]}, {expected[1]}, "
            f"{expected[2]}], got {tuple(hidden_states.shape)}."
        )


class PoolingOutputTransform(nn.Module):
    """Shared activation, optional RMSNorm, then dropout output stage."""

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__()
        self.activation = nn.SiLU() if config.output_act == "silu" else nn.Identity()
        self.norm = (
            RMSNorm(config.output_size, eps=config.layer_norm_eps)
            if config.use_output_norm
            else nn.Identity()
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.activation(pooled)))
