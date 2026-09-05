from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from .configuration import SSLFrontendConfig


@dataclass
class SSLFrontendOutput(ModelOutput):
    hidden_state: Optional[torch.Tensor] = None


class SSLFrontendSummaryMixin:
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
        output_projection = (
            f"Linear({self.config.ssl_hidden_size}->{self.config.output_size}, bias=False)"
            if isinstance(self.output_projection, nn.Linear)
            else "Identity"
        )
        return "\n".join(
            [
                "SSLFrontendModel",
                f"  ssl_type: {self.config.ssl_type}",
                f"  checkpoint: {self.config.pretrained_model_name_or_path}",
                f"  target_sampling_rate: {self.config.target_sampling_rate}",
                f"  normalize_input: {self.config.normalize_input}",
                f"  freeze_ssl_model: {self.config.freeze_ssl_model}",
                f"  tuning_mode: {self.ssl_tuning_mode}",
                f"  trainable_encoder_layers: {self.trainable_encoder_layer_indices}",
                f"  encoder: {self.ssl_model.__class__.__name__}",
                f"  ssl_hidden_size: {self.config.ssl_hidden_size}",
                f"  output_projection: {output_projection}",
                f"  output_size: {self.config.frontend_output_dim}",
            ]
        )

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class SSLFrontendModel(SSLFrontendSummaryMixin, PreTrainedModel):
    """Extract frame-level representations from raw 16 kHz waveforms."""

    config_class = SSLFrontendConfig
    base_model_prefix = "ssl_model"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    def __init__(self, config: SSLFrontendConfig) -> None:
        super().__init__(config)
        if torch.get_default_device().type == "meta":
            ssl_model_config = self._load_ssl_model_config(config)
            self.ssl_model = AutoModel.from_config(ssl_model_config)
        else:
            self.ssl_model = AutoModel.from_pretrained(
                config.pretrained_model_name_or_path,
            )
        config.ssl_model_config = self.ssl_model.config.to_dict()
        model_hidden_size = int(self.ssl_model.config.hidden_size)
        if model_hidden_size != config.ssl_hidden_size:
            raise ValueError(
                f"Loaded SSL model hidden_size={model_hidden_size} does not match "
                f"configured ssl_hidden_size={config.ssl_hidden_size}."
            )
        self._configure_ssl_trainability()
        self.output_projection = (
            nn.Linear(config.ssl_hidden_size, config.output_size, bias=False)
            if config.ssl_hidden_size != config.output_size
            else nn.Identity()
        )
        if isinstance(self.output_projection, nn.Linear):
            nn.init.normal_(
                self.output_projection.weight,
                mean=0.0,
                std=config.initializer_range,
            )

    def _get_encoder_layers(self) -> nn.ModuleList:
        encoder = getattr(self.ssl_model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError(
                f"{self.ssl_model.__class__.__name__} does not expose "
                "encoder.layers as nn.ModuleList."
            )
        return layers

    def _configure_ssl_trainability(self) -> None:
        selected_indices = self.config.ssl_trainable_layer_indices
        if selected_indices is None:
            if self.config.freeze_ssl_model:
                self.ssl_model.requires_grad_(False)
            self.trainable_encoder_layer_indices = None
            self.ssl_tuning_mode = (
                "frozen" if self.config.freeze_ssl_model else "full"
            )
            return

        encoder_layers = self._get_encoder_layers()
        num_layers = len(encoder_layers)
        invalid_indices = [
            index for index in selected_indices if index >= num_layers
        ]
        if invalid_indices:
            raise ValueError(
                f"ssl_trainable_layer_indices contains {invalid_indices}, but "
                f"{self.config.ssl_type} has {num_layers} encoder layers."
            )

        self.ssl_model.requires_grad_(False)
        feature_extractor = getattr(
            self.ssl_model,
            "feature_extractor",
            None,
        )
        freeze_parameters = getattr(
            feature_extractor,
            "_freeze_parameters",
            None,
        )
        if not callable(freeze_parameters):
            raise TypeError(
                f"{self.ssl_model.__class__.__name__} does not expose a "
                "compatible feature extractor freeze method."
            )
        freeze_parameters()
        for index in selected_indices:
            encoder_layers[index].requires_grad_(True)

        self.trainable_encoder_layer_indices = tuple(selected_indices)
        self.ssl_tuning_mode = "selected"

    @staticmethod
    def _load_ssl_model_config(config: SSLFrontendConfig) -> PretrainedConfig:
        if config.ssl_model_config is not None:
            config_dict = dict(config.ssl_model_config)
            model_type = config_dict.pop("model_type", None)
            if not isinstance(model_type, str) or not model_type:
                raise ValueError("ssl_model_config must contain a non-empty model_type.")
            return AutoConfig.for_model(model_type, **config_dict)

        # Legacy ADD checkpoints predate embedded SSL configs. Their architecture
        # config is read from the local HF cache; outer ADD weights are loaded next.
        return AutoConfig.from_pretrained(
            config.pretrained_model_name_or_path,
            local_files_only=True,
        )

    def train(self, mode: bool = True) -> SSLFrontendModel:
        super().train(mode)
        if self.ssl_tuning_mode == "frozen":
            self.ssl_model.eval()
        return self

    def _normalize_input(self, input_values: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_input:
            return input_values
        input_values_float = input_values.float()
        mean = input_values_float.mean(dim=-1, keepdim=True)
        variance = input_values_float.var(dim=-1, unbiased=False, keepdim=True)
        return (input_values_float - mean) * torch.rsqrt(
            variance + self.config.input_norm_eps
        )

    def _forward_ssl_encoder(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.ssl_model(
            input_values=input_values,
            return_dict=True,
        ).last_hidden_state

    def forward(
        self,
        input_values: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> tuple | SSLFrontendOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch_size, num_samples].")

        input_values = self._normalize_input(input_values)
        hidden_state = self.output_projection(self._forward_ssl_encoder(input_values))
        if hidden_state.is_cuda and torch.is_autocast_enabled("cuda"):
            hidden_state = hidden_state.to(torch.get_autocast_dtype("cuda"))

        if not return_dict:
            return (hidden_state,)
        return SSLFrontendOutput(hidden_state=hidden_state)


__all__ = [
    "SSLFrontendModel",
    "SSLFrontendOutput",
    "SSLFrontendSummaryMixin",
]
