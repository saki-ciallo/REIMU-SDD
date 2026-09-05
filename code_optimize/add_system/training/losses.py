from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..configuration.settings import LossSettings


class FocalLoss(nn.Module):
    """Numerically stable weighted multiclass focal loss."""

    def __init__(
        self,
        class_weights: tuple[float, ...],
        *,
        gamma: float,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits.float(), dim=-1)
        labels = labels.to(device=logits.device, dtype=torch.long)
        target_log_probability = log_probabilities.gather(
            dim=-1,
            index=labels[:, None],
        ).squeeze(-1)
        target_probability = target_log_probability.exp()
        alpha = self.class_weights.gather(0, labels)
        return (-alpha * (1.0 - target_probability).pow(self.gamma) * target_log_probability).mean()


def build_loss(settings: LossSettings, num_labels: int) -> nn.Module:
    if num_labels != 2:
        raise ValueError("bonafide/spoof weighting requires num_labels=2.")
    weights = torch.tensor(settings.class_weights, dtype=torch.float32)
    if settings.loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=settings.label_smoothing,
        )
    return FocalLoss(settings.class_weights, gamma=settings.focal_gamma)
