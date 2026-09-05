from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from ...configuration.components.frontends import SincNetFrontendConfig
from ..outputs import FrontendOutput


class SincConvFast(nn.Module):
    """Vectorized trainable band-pass filter bank."""

    def __init__(self, config: SincNetFrontendConfig) -> None:
        super().__init__()
        self.out_channels = config.num_filters
        requested_kernel = config.window_size
        self.kernel_size = requested_kernel + 1 if requested_kernel % 2 == 0 else requested_kernel
        self.sample_rate = config.target_sampling_rate
        self.stride = config.stride_size
        self.min_low_hz = config.min_low_hz
        self.min_band_hz = config.min_band_hz

        low_hz, band_hz = self._mel_initialization()
        self.low_hz_ = nn.Parameter(low_hz[:, None])
        self.band_hz_ = nn.Parameter(band_hz[:, None])

    @staticmethod
    def _to_mel(hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    @staticmethod
    def _to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _mel_initialization(self) -> tuple[torch.Tensor, torch.Tensor]:
        low_hz = torch.tensor(30.0, dtype=torch.float32)
        high_hz = torch.tensor(
            self.sample_rate / 2.0 - (self.min_low_hz + self.min_band_hz),
            dtype=torch.float32,
        )
        mel = torch.linspace(
            self._to_mel(low_hz),
            self._to_mel(high_hz),
            self.out_channels + 1,
        )
        hz = self._to_hz(mel)
        return hz[:-1], torch.diff(hz)

    def filters(self) -> torch.Tensor:
        # Kernel generation stays in FP32 for stable cutoff-frequency arithmetic.
        with torch.autocast(device_type=self.low_hz_.device.type, enabled=False):
            low = self.min_low_hz + self.low_hz_.float().abs()
            high = torch.clamp(
                low + self.min_band_hz + self.band_hz_.float().abs(),
                self.min_low_hz,
                self.sample_rate / 2.0,
            )
            band = (high - low)[:, 0]
            half_kernel = self.kernel_size // 2
            n = torch.arange(half_kernel, device=low.device, dtype=low.dtype)
            window = 0.54 - 0.46 * torch.cos(2.0 * torch.pi * n / self.kernel_size)
            negative_time = (
                2.0
                * torch.pi
                * torch.arange(
                    -half_kernel,
                    0,
                    device=low.device,
                    dtype=low.dtype,
                )[None, :]
                / self.sample_rate
            )
            left = (
                (torch.sin(high @ negative_time) - torch.sin(low @ negative_time))
                / (negative_time / 2.0)
            ) * window
            center = 2.0 * band[:, None]
            filters = torch.cat((left, center, left.flip(1)), dim=1)
            filters = filters / (2.0 * band[:, None])
            return filters[:, None, :]

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch, samples].")
        return F.conv1d(
            input_values[:, None, :],
            self.filters(),
            stride=self.stride,
        )


class CausalConvBlock(nn.Module):
    """Causal Conv1d, channel RMSNorm, and optional SiLU."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        *,
        bias: bool,
        norm_eps: float,
        apply_activation: bool,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.convolution = nn.Conv1d(
            input_size,
            output_size,
            kernel_size,
            bias=bias,
        )
        self.norm = nn.RMSNorm(output_size, eps=norm_eps)
        self.activation = nn.SiLU() if apply_activation else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (self.kernel_size - 1, 0))
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        return self.activation(hidden_states)


class SincCNNEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        kernels: Sequence[int],
        channels: Sequence[int],
        config: SincNetFrontendConfig,
    ) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        current_size = input_size
        for index, (kernel, next_size) in enumerate(zip(kernels, channels, strict=True)):
            is_last = index == len(kernels) - 1
            stages.append(
                CausalConvBlock(
                    current_size,
                    next_size,
                    kernel,
                    bias=config.cnn_bias,
                    norm_eps=config.rms_norm_eps,
                    apply_activation=(config.apply_final_cnn_activation if is_last else True),
                )
            )
            current_size = next_size
        self.stages = nn.ModuleList(stages)
        self.output_size = current_size
        self.residual_projection = (
            nn.Conv1d(input_size, current_size, 1, bias=config.cnn_bias)
            if config.use_final_sinc_residual
            else None
        )
        self.output_activation = nn.SiLU()
        self.dropout = nn.Dropout(config.final_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        for stage in self.stages:
            hidden_states = stage(hidden_states)
        if self.residual_projection is not None:
            hidden_states = hidden_states + self.residual_projection(residual)
        return self.dropout(self.output_activation(hidden_states))


class SincNetFrontendModel(PreTrainedModel):
    config_class = SincNetFrontendConfig
    base_model_prefix = "frontend"
    main_input_name = "input_values"
    _no_split_modules: ClassVar[list[str]] = ["SincConvFast", "CausalConvBlock"]

    def __init__(self, config: SincNetFrontendConfig) -> None:
        super().__init__(config)
        self.sinc = SincConvFast(config)
        self.encoder = SincCNNEncoder(
            config.num_filters,
            config.cnn_kernels,
            config.cnn_channels,
            config,
        )
        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | FrontendOutput:
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        input_values = input_values.to(dtype=self.sinc.low_hz_.dtype)
        hidden_state = self.encoder(self.sinc(input_values)).transpose(1, 2).contiguous()
        if torch.is_autocast_enabled(input_values.device.type):
            hidden_state = hidden_state.to(torch.get_autocast_dtype(input_values.device.type))
        if not use_return_dict:
            return (hidden_state,)
        return FrontendOutput(hidden_state=hidden_state)
