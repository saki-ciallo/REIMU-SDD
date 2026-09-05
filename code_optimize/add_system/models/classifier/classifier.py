from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn
from transformers import PreTrainedModel

from ...configuration.components.classifier import LinearClassifierConfig
from ..outputs import ClassifierOutput
from .amsoftmax import AMSoftmaxClassifier
from .head import ClassifierHead


class ClassifierModel(PreTrainedModel):
    config_class = LinearClassifierConfig
    base_model_prefix = "classifier"
    _no_split_modules: ClassVar[list[str]] = [
        "ClassifierHead",
        "AMSoftmaxClassifier",
    ]

    def __init__(self, config: LinearClassifierConfig) -> None:
        super().__init__(config)
        self.classifier = ClassifierHead(config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, AMSoftmaxClassifier):
            nn.init.xavier_normal_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, ...] | ClassifierOutput:
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        logits = self.classifier(input_ids)
        loss_logits = None
        projection = self.classifier.projection
        if labels is not None and isinstance(projection, AMSoftmaxClassifier):
            loss_logits = projection.apply_margin(logits, labels)
        if not use_return_dict:
            return tuple(value for value in (logits, loss_logits) if value is not None)
        return ClassifierOutput(logits=logits, loss_logits=loss_logits)
