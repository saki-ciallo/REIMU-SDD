from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...configuration.components.pooling import GatedAttentionPoolingConfig
from .base import PoolingOutputTransform, validate_pooling_input


class MultiHeadGatedAttentionPooling(nn.Module):
    """Per-head value scoring and gated evidence aggregation."""

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.num_heads = config.resolved_num_heads
        self.head_dim = config.resolved_head_dim
        self.score_temperature = config.score_temperature
        self.value_gate_projection = nn.Linear(
            self.input_size,
            2 * self.output_size,
            bias=config.bias,
        )
        self.score_weight = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.gate_activation = torch.sigmoid if config.gate_act == "sigmoid" else F.silu
        self.output_transform = PoolingOutputTransform(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        validate_pooling_input(hidden_states, self.input_size)
        batch_size, sequence_length, _ = hidden_states.shape
        values, gates = self.value_gate_projection(hidden_states).chunk(2, dim=-1)
        head_shape = (
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        values = values.reshape(head_shape)
        evidence = values * self.gate_activation(gates.reshape(head_shape))

        logits = torch.einsum("bshd,hd->bsh", values, self.score_weight)
        weights = torch.softmax(
            logits.float() / self.score_temperature,
            dim=1,
        ).to(evidence.dtype)
        pooled = torch.einsum("bsh,bshd->bhd", weights, evidence)
        return self.output_transform(pooled.flatten(1))
