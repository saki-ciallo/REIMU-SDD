from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = F.softmax(
            F.linear(hidden_states, self.weight).float(),
            dim=-1,
        )
        top_weights, top_indices = probabilities.topk(self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probabilities, top_weights, top_indices
