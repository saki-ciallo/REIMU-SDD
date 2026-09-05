from __future__ import annotations

import torch
from torch import nn

from ...configuration.components.classifier import ClassifierType, LinearClassifierConfig
from .amsoftmax import AMSoftmaxClassifier


class ClassifierHead(nn.Module):
    def __init__(self, config: LinearClassifierConfig) -> None:
        super().__init__()
        self.input_size = config.input_size
        self.classifier_type = ClassifierType(config.classifier_type)
        if self.classifier_type is ClassifierType.LINEAR:
            self.projection: nn.Module = nn.Linear(
                config.input_size,
                config.num_labels,
                bias=config.bias,
            )
        else:
            self.projection = AMSoftmaxClassifier(
                config.input_size,
                config.num_labels,
                margin=config.am_margin,
                scale=config.am_scale,
                eps=config.am_eps,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.input_size:
            raise ValueError(
                f"hidden_states must have shape [batch, {self.input_size}], "
                f"got {tuple(hidden_states.shape)}."
            )
        return self.projection(hidden_states)
