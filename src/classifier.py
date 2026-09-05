from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import LinearClassifierConfig


class AMSoftmaxClassifier(nn.Module):
    """Additive-margin softmax classifier head that returns logits only."""

    def __init__(
        self,
        input_size: int,
        num_labels: int,
        margin: float,
        scale: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.margin = margin
        self.scale = scale
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(num_labels, input_size))
        nn.init.xavier_normal_(self.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            features = F.normalize(hidden_states.float(), p=2, dim=1, eps=self.eps)
            weight = F.normalize(self.weight.float(), p=2, dim=1, eps=self.eps)
            logits = F.linear(features, weight)

            if labels is not None:
                labels = labels.to(device=logits.device, dtype=torch.long).view(-1)
                if labels.shape[0] != logits.shape[0]:
                    raise ValueError(
                        f"labels batch size must match hidden_states batch size, "
                        f"got {labels.shape[0]} and {logits.shape[0]}."
                    )
                logits = logits.clone()
                row_indices = torch.arange(logits.shape[0], device=logits.device)
                logits[row_indices, labels] = logits[row_indices, labels] - self.margin

            return logits * self.scale


class LinearClassifier(nn.Module):
    """Classifier for pooled features."""

    def __init__(self, config: LinearClassifierConfig) -> None:
        super().__init__()

        if config.input_size <= 0:
            raise ValueError("input_size must be positive.") 
        if config.num_labels <= 0:
            raise ValueError("num_labels must be positive.")

        self.input_size = config.input_size
        self.num_labels = config.num_labels
        self.classifier_type = config.classifier_type
        if self.classifier_type == "linear":
            self.classifier = nn.Linear(self.input_size, self.num_labels, bias=config.bias)
        elif self.classifier_type == "amsoftmax":
            self.classifier = AMSoftmaxClassifier(
                input_size=self.input_size,
                num_labels=self.num_labels,
                margin=config.am_margin,
                scale=config.am_scale,
                eps=config.am_eps,
            )
        else:
            raise ValueError(
                f"classifier_type must be one of linear or amsoftmax, got {self.classifier_type!r}."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [batch_size, input_size].")
        if hidden_states.shape[-1] != self.input_size:
            raise ValueError(
                f"hidden_states last dimension must match input_size={self.input_size}, "
                f"got {hidden_states.shape[-1]}."
            )

        if self.classifier_type == "amsoftmax":
            return self.classifier(hidden_states, labels=labels)
        return self.classifier(hidden_states)


__all__ = [
    "AMSoftmaxClassifier",
    "LinearClassifier",
]
