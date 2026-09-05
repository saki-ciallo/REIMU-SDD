from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn
from transformers import PreTrainedModel

from ...configuration.components.frontends import LinearFrontendConfig
from ..outputs import FrontendOutput


class LinearAudioFrontend(nn.Module):
    """Slice complete, non-overlapping frames and project them to model width."""

    def __init__(self, config: LinearFrontendConfig) -> None:
        super().__init__()
        self.frame_size = config.frame_size
        self.output_size = config.output_size
        self.projection = nn.Linear(
            self.frame_size,
            self.output_size,
            bias=config.bias,
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch, samples].")
        num_frames = input_values.shape[-1] // self.frame_size
        if num_frames == 0:
            raise ValueError(f"input_values must contain at least {self.frame_size} samples.")
        frames = input_values[:, : num_frames * self.frame_size].unflatten(
            -1,
            (num_frames, self.frame_size),
        )
        return self.projection(frames)


class LinearFrontendModel(PreTrainedModel):
    config_class = LinearFrontendConfig
    base_model_prefix = "frontend"
    main_input_name = "input_values"
    _no_split_modules: ClassVar[list[str]] = ["LinearAudioFrontend"]

    def __init__(self, config: LinearFrontendConfig) -> None:
        super().__init__(config)
        self.frontend = LinearAudioFrontend(config)
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

    def forward(
        self,
        input_values: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | FrontendOutput:
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        hidden_state = self.frontend(input_values)
        if not use_return_dict:
            return (hidden_state,)
        return FrontendOutput(hidden_state=hidden_state)
