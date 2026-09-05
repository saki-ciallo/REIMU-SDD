from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration import SincNetFrontendConfig
from .frontend import MixedCNN, SincConv_fast, StreamingMixedCNNState, StreamingSincState


@dataclass
class SincNetBlockStreamingState:
    sinc_state: StreamingSincState = field(default_factory=StreamingSincState)
    cnns_state: StreamingMixedCNNState = field(default_factory=StreamingMixedCNNState)


class SincNetBlock(nn.Module):
    """Reusable SincConv + causal CNN frontend block."""

    def __init__(self, config: SincNetFrontendConfig) -> None:
        super().__init__()
        self.sincnet = SincConv_fast(
            out_channels=config.num_filters,
            kernel_size=config.window_size,
            sample_rate=config.target_sampling_rate,
            stride=config.stride_size,
            min_low_hz=config.min_low_hz,
            min_band_hz=config.min_band_hz,
        )
        self.cnns = MixedCNN(
            input_dim=config.num_filters,
            kernels=config.cnn_kernels,
            channels=config.cnn_channels,
            bias=config.cnn_bias,
            apply_final_cnn_activation=config.apply_final_cnn_activation,
            use_final_sinc_residual=config.use_final_sinc_residual,
            final_dropout=config.final_dropout,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.output_dim = self.cnns.output_dim

    def init_streaming_state(self) -> SincNetBlockStreamingState:
        return SincNetBlockStreamingState(
            sinc_state=self.sincnet.reset_streaming_state(), # StreamingSincState
            cnns_state=self.cnns.reset_streaming_state(), # StreamingMixedCNNState
        )

    def reset_streaming_state(self) -> SincNetBlockStreamingState:
        return self.init_streaming_state()

    def forward(self, input_samples: torch.Tensor) -> torch.Tensor:
        sinc_features = self.sincnet(input_samples)
        return self.cnns(sinc_features).transpose(1, 2).contiguous()

    def forward_with_sinc_features(self, input_samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 用于分析 sinc 的特征形状
        sinc_features = self.sincnet(input_samples)
        hidden_states = self.cnns(sinc_features).transpose(1, 2).contiguous()
        return hidden_states, sinc_features

    def forward_streaming(
        self,
        new_input_samples: torch.Tensor,
        state: Optional[SincNetBlockStreamingState] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], SincNetBlockStreamingState]:
        # 流输入用，sincnet 和 cnns 都实现了对应的 forward_streaming，直接调用
        state = self.reset_streaming_state() if state is None else state
        if not state.cnns_state.layer_states:
            state.cnns_state = self.cnns.init_streaming_state()
        sinc_features, sinc_state = self.sincnet.forward_streaming(
            new_input_samples, # 对应 raw audio samples
            state.sinc_state,
        )
        state.sinc_state = sinc_state
        if sinc_features is None: # 倘若输入极短小于 kenel_size 则不能正常卷积
            return None, None, state # 直接将累积的状态返回，不需要到 cnns 了

        hidden_states, cnns_state = self.cnns.forward_streaming(
            sinc_features,
            state.cnns_state,
        )
        state.cnns_state = cnns_state
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        return hidden_states, sinc_features, state


@dataclass
class SincNetFrontendOutput(ModelOutput):
    hidden_state: Optional[torch.Tensor] = None


class SincNetFrontendSummaryMixin:
    def trainable_parameter_summary(self) -> List[Dict[str, object]]:
        descriptions = {
            "low_hz_": "lower cutoff in Hz per SincNet filter",
            "band_hz_": "bandwidth in Hz per SincNet filter",
        }
        return [
            {
                "name": name,
                "shape": tuple(parameter.shape),
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "numel": parameter.numel(),
                "meaning": descriptions.get(name.rsplit(".", 1)[-1], "frontend/CNN trainable parameter"),
            }
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def sinc_parameter_summary(self) -> List[Dict[str, object]]:
        return [
            item
            for item in self.trainable_parameter_summary()
            if item["name"].endswith(("sincnet.low_hz_", "sincnet.band_hz_"))
        ]

    def cnns_parameter_summary(self) -> List[Dict[str, object]]:
        summaries = []
        for layer_idx, layer in enumerate(self.frontend.cnns.layers):
            parameter_items = []
            layer_prefix = f"frontend.cnns.layers.{layer_idx}"
            for parameter_name, parameter in layer.named_parameters():
                if not parameter.requires_grad:
                    continue
                full_name = f"{layer_prefix}.{parameter_name}"
                parameter_items.append(
                    {
                        "name": full_name,
                        "shape": tuple(parameter.shape),
                        "dtype": str(parameter.dtype).replace("torch.", ""),
                        "numel": parameter.numel(),
                        "meaning": self._cnn_parameter_meaning(parameter_name),
                    }
                )
            summaries.append(
                {
                    "layer_index": layer_idx,
                    "layer_name": f"layers.{layer_idx}",
                    "module": layer.__class__.__name__,
                    "input_dim": layer.input_dim,
                    "output_dim": layer.output_dim,
                    "kernel_size": layer.kernel_size,
                    "numel": sum(item["numel"] for item in parameter_items),
                    "parameters": parameter_items,
                }
            )

        final_residual_proj = self.frontend.cnns.final_residual_proj
        if final_residual_proj is not None:
            parameter_items = []
            for parameter_name, parameter in final_residual_proj.named_parameters():
                if not parameter.requires_grad:
                    continue
                full_name = f"frontend.cnns.final_residual_proj.{parameter_name}"
                parameter_items.append(
                    {
                        "name": full_name,
                        "shape": tuple(parameter.shape),
                        "dtype": str(parameter.dtype).replace("torch.", ""),
                        "numel": parameter.numel(),
                        "meaning": self._cnn_parameter_meaning(parameter_name),
                    }
                )
            summaries.append(
                {
                    "layer_index": None,
                    "layer_name": "final_residual_proj",
                    "module": final_residual_proj.__class__.__name__,
                    "input_dim": self.config.num_filters,
                    "output_dim": self.frontend.output_dim,
                    "kernel_size": 1,
                    "numel": sum(item["numel"] for item in parameter_items),
                    "parameters": parameter_items,
                }
            )

        return summaries

    @staticmethod
    def _cnn_parameter_meaning(parameter_name: str) -> str:
        if parameter_name == "conv.weight":
            return "causal CNN convolution weight"
        if parameter_name == "conv.bias":
            return "causal CNN convolution bias"
        if parameter_name == "norm.weight":
            return "RMSNorm scale for CNN channel dimension"
        if parameter_name == "weight":
            return "linear projection weight"
        if parameter_name == "bias":
            return "linear projection bias"
        return "CNN trainable parameter"

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
        lines = [
            "SincNetFrontendModel",
            (
                "  frontend.sincnet: "
                f"SincConv_fast(out_channels={self.frontend.sincnet.out_channels}, "
                f"kernel_size={self.frontend.sincnet.kernel_size}, stride={self.frontend.sincnet.stride})"
            ),
            f"  frontend.cnns: MixedCNN(num_layers={len(self.frontend.cnns.layers)})",
        ]
        for layer_idx, layer in enumerate(self.frontend.cnns.layers):
            lines.extend(
                [
                    (
                        f"    layers.{layer_idx}: CausalCNN("
                        f"{layer.input_dim}->{layer.output_dim}, kernel={layer.kernel_size})"
                    ),
                    f"      conv: {layer.conv.__class__.__name__}",
                    f"      norm: {layer.norm.__class__.__name__}",
                    f"      activation: {layer.activation.__class__.__name__ if layer.apply_activation else 'deferred-final-SiLU'}",
                ]
            )
        final_residual = (
            self.frontend.cnns.final_residual_proj.__class__.__name__
            if self.frontend.cnns.final_residual_proj is not None
            else "None"
        )
        lines.append(f"  frontend.cnns.final_residual_proj: {final_residual}")
        lines.append(f"  frontend.cnns.final_activation: {self.frontend.cnns.final_activation.__class__.__name__}")
        final_dropout = self.frontend.cnns.final_dropout
        final_dropout_p = final_dropout.p if isinstance(final_dropout, nn.Dropout) else 0.0
        lines.append(
            f"  frontend.cnns.final_dropout: {final_dropout.__class__.__name__}(p={final_dropout_p})"
        )
        return "\n".join(lines)

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class SincNetFrontendModel(SincNetFrontendSummaryMixin, PreTrainedModel):
    config_class = SincNetFrontendConfig
    base_model_prefix = "frontend"
    main_input_name = "input_values"
    _no_split_modules = ["SincNetBlock"]

    def __init__(self, config: SincNetFrontendConfig) -> None:
        super().__init__(config)
        self.frontend = SincNetBlock(config)
        self.post_init()

    def init_streaming_state(self) -> SincNetBlockStreamingState:
        return self.frontend.init_streaming_state()

    def reset_streaming_state(self) -> SincNetBlockStreamingState:
        return self.init_streaming_state()

    def forward(
        self,
        input_values: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> tuple | SincNetFrontendOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_values.ndim != 2: # 输入 [B, Samples]
            raise ValueError("input_values must have shape [batch_size, num_samples].")

        input_values = input_values.to(dtype=self.frontend.sincnet.low_hz_.dtype)
        hidden_states = self.frontend(input_values)  # 输出 [B, T, C]
        if torch.is_autocast_enabled(input_values.device.type):
            hidden_states = hidden_states.to(
                dtype=torch.get_autocast_dtype(input_values.device.type)
            )

        if not return_dict:
            return (hidden_states,)
        return SincNetFrontendOutput(
            hidden_state=hidden_states,
        )

    def forward_streaming(
        self,
        new_input_samples: torch.Tensor,
        state: Optional[SincNetBlockStreamingState] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], SincNetBlockStreamingState]:
        new_input_samples = new_input_samples.to(dtype=self.frontend.sincnet.low_hz_.dtype)
        return self.frontend.forward_streaming(new_input_samples, state)
