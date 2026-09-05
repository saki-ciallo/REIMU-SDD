from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration import LinearFrontendConfig
from .frontend import LinearAudioFrontend


@dataclass
class LinearFrontendOutput(ModelOutput):
    hidden_state: Optional[torch.Tensor] = None


class LinearFrontendSummaryMixin:
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
        frontend = self.frontend
        return "\n".join(
            [
                "LinearFrontendModel",
                (
                    "  frontend: "
                    f"LinearAudioFrontend(frame_size={frontend.frame_size}, "
                    f"output_size={frontend.output_size})"
                ),
                f"    proj: Linear({frontend.frame_size}->{frontend.output_size})",
            ]
        )

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class LinearFrontendModel(LinearFrontendSummaryMixin, PreTrainedModel):
    config_class = LinearFrontendConfig
    base_model_prefix = "frontend"
    main_input_name = "input_values"
    _no_split_modules = ["LinearAudioFrontend"]

    def __init__(self, config: LinearFrontendConfig) -> None:
        super().__init__(config)
        self.frontend = LinearAudioFrontend(
            frame_size=config.frame_size,
            output_size=config.output_size,
            bias=config.bias,
        )
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_values: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> tuple | LinearFrontendOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        hidden_state = self.frontend(input_values=input_values)

        if not return_dict:
            return (hidden_state,)

        return LinearFrontendOutput(hidden_state=hidden_state)


__all__ = [
    "LinearFrontendModel",
    "LinearFrontendOutput",
    "LinearFrontendSummaryMixin",
]
