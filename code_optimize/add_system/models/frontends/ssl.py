from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from ...configuration.components.frontends import SSLFrontendConfig
from ..outputs import FrontendOutput


class SSLFrontendModel(PreTrainedModel):
    """Raw-waveform SSL encoder with full, frozen, or selected-layer tuning."""

    config_class = SSLFrontendConfig
    base_model_prefix = "ssl_model"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    def __init__(self, config: SSLFrontendConfig) -> None:
        super().__init__(config)
        if torch.get_default_device().type == "meta":
            self.ssl_model = AutoModel.from_config(self._embedded_ssl_config(config))
        else:
            self.ssl_model = AutoModel.from_pretrained(
                config.pretrained_model_name_or_path,
                local_files_only=config.local_files_only,
            )
        config.ssl_model_config = self.ssl_model.config.to_dict()
        model_width = int(self.ssl_model.config.hidden_size)
        if model_width != config.ssl_hidden_size:
            raise ValueError(
                f"Loaded SSL width is {model_width}; expected {config.ssl_hidden_size}."
            )
        self._configure_trainability()
        self.output_projection = (
            nn.Linear(model_width, config.output_size, bias=False)
            if model_width != config.output_size
            else nn.Identity()
        )
        if isinstance(self.output_projection, nn.Linear):
            nn.init.normal_(
                self.output_projection.weight,
                mean=0.0,
                std=config.initializer_range,
            )

    @staticmethod
    def _embedded_ssl_config(config: SSLFrontendConfig) -> PretrainedConfig:
        if config.ssl_model_config is None:
            return AutoConfig.from_pretrained(
                config.pretrained_model_name_or_path,
                local_files_only=config.local_files_only,
            )
        values = dict(config.ssl_model_config)
        model_type = values.pop("model_type", None)
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("ssl_model_config must contain model_type.")
        return AutoConfig.for_model(model_type, **values)

    def _encoder_layers(self) -> nn.ModuleList:
        layers = getattr(getattr(self.ssl_model, "encoder", None), "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("SSL model must expose encoder.layers as nn.ModuleList.")
        return layers

    def _configure_trainability(self) -> None:
        selected = self.config.ssl_trainable_layer_indices
        if selected is None:
            if self.config.freeze_ssl_model:
                self.ssl_model.requires_grad_(False)
                self.ssl_tuning_mode = "frozen"
            else:
                self.ssl_tuning_mode = "full"
            self.trainable_encoder_layer_indices = None
            return

        layers = self._encoder_layers()
        invalid = [index for index in selected if index >= len(layers)]
        if invalid:
            raise ValueError(f"SSL layer indices {invalid} exceed encoder depth {len(layers)}.")
        self.ssl_model.requires_grad_(False)
        for index in selected:
            layers[index].requires_grad_(True)
        self.trainable_encoder_layer_indices = tuple(selected)
        self.ssl_tuning_mode = "selected"

    def train(self, mode: bool = True) -> SSLFrontendModel:
        super().train(mode)
        if self.ssl_tuning_mode == "frozen":
            self.ssl_model.eval()
        return self

    def _normalize_input(self, input_values: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_input:
            return input_values
        values = input_values.float()
        variance, mean = torch.var_mean(values, dim=-1, unbiased=False, keepdim=True)
        return (values - mean) * torch.rsqrt(variance + self.config.input_norm_eps)

    def forward(
        self,
        input_values: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | FrontendOutput:
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch, samples].")
        encoded = self.ssl_model(
            input_values=self._normalize_input(input_values),
            return_dict=True,
        ).last_hidden_state
        hidden_state = self.output_projection(encoded)
        if hidden_state.is_cuda and torch.is_autocast_enabled("cuda"):
            hidden_state = hidden_state.to(torch.get_autocast_dtype("cuda"))
        if not use_return_dict:
            return (hidden_state,)
        return FrontendOutput(hidden_state=hidden_state)
