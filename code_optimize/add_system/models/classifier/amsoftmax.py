from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AMSoftmaxClassifier(nn.Module):
    """Normalized additive-margin classifier that returns loss-agnostic logits."""

    def __init__(
        self,
        input_size: int,
        num_labels: int,
        *,
        margin: float,
        scale: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(num_labels, input_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            features = F.normalize(hidden_states.float(), dim=1, eps=self.eps)
            weight = F.normalize(self.weight.float(), dim=1, eps=self.eps)
            return F.linear(features, weight) * self.scale

    def apply_margin(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.to(device=logits.device, dtype=torch.long).flatten()
        if labels.shape[0] != logits.shape[0]:
            raise ValueError("labels and hidden_states batch sizes must match.")
        margins = F.one_hot(
            labels,
            num_classes=logits.shape[-1],
        ).to(logits.dtype)
        return logits - margins * (self.margin * self.scale)
