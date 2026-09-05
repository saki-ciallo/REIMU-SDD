from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .classifier import AMSoftmaxClassifier, LinearClassifier
from .configuration import LinearClassifierConfig


@dataclass
class LinearClassifierOutput(ModelOutput):
    logits: Optional[torch.Tensor] = None


class LinearClassifierSummaryMixin:
    def parameter_name_summary(self) -> List[Dict[str, object]]:
        return [
            {
                "name": name,
                "shape": tuple(parameter.shape),
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in self.named_parameters()
        ]

    def architecture_summary(self) -> str:
        classifier = self.classifier
        head = classifier.classifier
        if isinstance(head, AMSoftmaxClassifier):
            head_summary = (
                "AMSoftmaxClassifier("
                f"input_size={head.input_size}, "
                f"num_labels={head.num_labels}, "
                f"margin={head.margin}, "
                f"scale={head.scale})"
            )
        else:
            head_summary = f"Linear({classifier.input_size}->{classifier.num_labels})"
        return "\n".join(
            [
                "LinearClassifierModel",
                (
                    "  classifier: "
                    f"LinearClassifier(type={classifier.classifier_type}, "
                    f"input_size={classifier.input_size}, "
                    f"num_labels={classifier.num_labels})"
                ),
                f"    classifier: {head_summary}",
            ]
        )

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class LinearClassifierModel(LinearClassifierSummaryMixin, PreTrainedModel):
    config_class = LinearClassifierConfig
    base_model_prefix = "classifier"
    _no_split_modules = ["LinearClassifier", "AMSoftmaxClassifier"]

    def __init__(self, config: LinearClassifierConfig) -> None:
        super().__init__(config)
        self.classifier = LinearClassifier(config)
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
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple | LinearClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        hidden_states = input_ids
        logits = self.classifier(hidden_states, labels=labels)

        if not return_dict:
            return (logits,)

        return LinearClassifierOutput(
            logits=logits,
        )


__all__ = [
    "LinearClassifierModel",
    "LinearClassifierOutput",
    "LinearClassifierSummaryMixin",
]
