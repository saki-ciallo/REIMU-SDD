from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from transformers import PretrainedConfig

from ..validation import (
    boolean,
    positive_float,
    positive_int,
    probability,
)


class SincNetFrontendConfig(PretrainedConfig):
    """Waveform-to-sequence configuration for the SincNet frontend."""

    model_type = "sincnet_frontend"

    def __init__(
        self,
        target_sampling_rate: int = 16_000,
        window_ms: float = 25.0,
        stride_ms: float = 20.0,
        num_filters: int = 80,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
        cnn_kernels: Sequence[int] | None = None,
        cnn_channels: Sequence[int] | None = None,
        cnn_bias: bool = False,
        apply_final_cnn_activation: bool = False,
        use_final_sinc_residual: bool = False,
        final_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        kernels = tuple(cnn_kernels or (5, 3))
        channels = tuple(cnn_channels or (128, 192))
        if len(kernels) != len(channels):
            raise ValueError("cnn_kernels and cnn_channels must contain the same number of stages.")

        self.target_sampling_rate = positive_int(
            target_sampling_rate,
            field_name="target_sampling_rate",
        )
        self.window_ms = positive_float(window_ms, field_name="window_ms")
        self.stride_ms = positive_float(stride_ms, field_name="stride_ms")
        self.num_filters = positive_int(num_filters, field_name="num_filters")
        self.min_low_hz = positive_float(min_low_hz, field_name="min_low_hz")
        self.min_band_hz = positive_float(min_band_hz, field_name="min_band_hz")
        self.cnn_kernels = [
            positive_int(value, field_name=f"cnn_kernels[{index}]")
            for index, value in enumerate(kernels)
        ]
        self.cnn_channels = [
            positive_int(value, field_name=f"cnn_channels[{index}]")
            for index, value in enumerate(channels)
        ]
        self.cnn_bias = boolean(cnn_bias, field_name="cnn_bias")
        self.apply_final_cnn_activation = boolean(
            apply_final_cnn_activation,
            field_name="apply_final_cnn_activation",
        )
        self.use_final_sinc_residual = boolean(
            use_final_sinc_residual,
            field_name="use_final_sinc_residual",
        )
        self.final_dropout = probability(final_dropout, field_name="final_dropout")
        self.rms_norm_eps = positive_float(rms_norm_eps, field_name="rms_norm_eps")

        if self.stride_size > self.window_size:
            raise ValueError("stride_ms must not produce a stride larger than window_ms.")

    @property
    def window_size(self) -> int:
        return round(self.target_sampling_rate * self.window_ms / 1000.0)

    @property
    def stride_size(self) -> int:
        return round(self.target_sampling_rate * self.stride_ms / 1000.0)

    @property
    def overlap_size(self) -> int:
        return self.window_size - self.stride_size

    @property
    def frontend_output_dim(self) -> int:
        return self.cnn_channels[-1]


class LinearFrontendConfig(PretrainedConfig):
    """Non-overlapping waveform slicing followed by a learned projection."""

    model_type = "linear_frontend"

    def __init__(
        self,
        target_sampling_rate: int = 16_000,
        frame_ms: float = 25.0,
        output_size: int = 512,
        bias: bool = False,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target_sampling_rate = positive_int(
            target_sampling_rate,
            field_name="target_sampling_rate",
        )
        self.frame_ms = positive_float(frame_ms, field_name="frame_ms")
        self.output_size = positive_int(output_size, field_name="output_size")
        self.bias = boolean(bias, field_name="bias")
        self.initializer_range = positive_float(
            initializer_range,
            field_name="initializer_range",
        )

    @property
    def frame_size(self) -> int:
        return round(self.target_sampling_rate * self.frame_ms / 1000.0)

    @property
    def frontend_output_dim(self) -> int:
        return self.output_size


class SSLFrontendConfig(PretrainedConfig):
    """Supported Hugging Face self-supervised speech frontend."""

    model_type = "ssl_frontend"
    SPECS: ClassVar[dict[str, dict[str, object]]] = {
        "wav2vec2-base": {
            "pretrained_model_name_or_path": "facebook/wav2vec2-base",
            "hidden_size": 768,
            "sampling_rate": 16_000,
            "normalize_input": True,
        },
        "hubert-base-ls960": {
            "pretrained_model_name_or_path": "facebook/hubert-base-ls960",
            "hidden_size": 768,
            "sampling_rate": 16_000,
            "normalize_input": True,
        },
        "wavlm-base": {
            "pretrained_model_name_or_path": "microsoft/wavlm-base",
            "hidden_size": 768,
            "sampling_rate": 16_000,
            "normalize_input": False,
        },
        "wavlm-base-plus": {
            "pretrained_model_name_or_path": "microsoft/wavlm-base-plus",
            "hidden_size": 768,
            "sampling_rate": 16_000,
            "normalize_input": False,
        },
    }

    def __init__(
        self,
        ssl_type: str = "wav2vec2-base",
        target_sampling_rate: int = 16_000,
        output_size: int = 768,
        input_norm_eps: float = 1e-7,
        freeze_ssl_model: bool = False,
        ssl_trainable_layer_indices: Sequence[int] | None = None,
        ssl_model_config: Mapping[str, object] | None = None,
        local_files_only: bool = False,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if ssl_type not in self.SPECS:
            supported = ", ".join(self.SPECS)
            raise ValueError(f"ssl_type must be one of {supported}, got {ssl_type!r}.")
        spec = self.SPECS[ssl_type]
        sampling_rate = positive_int(
            target_sampling_rate,
            field_name="target_sampling_rate",
        )
        expected_sampling_rate = int(spec["sampling_rate"])
        if sampling_rate != expected_sampling_rate:
            raise ValueError(
                f"{ssl_type} expects target_sampling_rate={expected_sampling_rate}, "
                f"got {sampling_rate}."
            )

        freeze = boolean(freeze_ssl_model, field_name="freeze_ssl_model")
        indices: list[int] | None = None
        if ssl_trainable_layer_indices is not None:
            if not freeze:
                raise ValueError("ssl_trainable_layer_indices requires freeze_ssl_model=True.")
            indices = []
            for index in ssl_trainable_layer_indices:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError("ssl_trainable_layer_indices must contain only integers.")
                if index < 0:
                    raise ValueError("ssl_trainable_layer_indices must be non-negative.")
                indices.append(index)
            if not indices:
                raise ValueError("ssl_trainable_layer_indices must not be empty.")
            if len(indices) != len(set(indices)):
                raise ValueError("ssl_trainable_layer_indices must not contain duplicates.")

        self.ssl_type = ssl_type
        self.pretrained_model_name_or_path = str(spec["pretrained_model_name_or_path"])
        self.target_sampling_rate = sampling_rate
        self.ssl_hidden_size = int(spec["hidden_size"])
        self.output_size = positive_int(output_size, field_name="output_size")
        self.normalize_input = bool(spec["normalize_input"])
        self.input_norm_eps = positive_float(input_norm_eps, field_name="input_norm_eps")
        self.freeze_ssl_model = freeze
        self.ssl_trainable_layer_indices = indices
        self.ssl_model_config = dict(ssl_model_config) if ssl_model_config is not None else None
        self.local_files_only = boolean(
            local_files_only,
            field_name="local_files_only",
        )
        self.initializer_range = positive_float(
            initializer_range,
            field_name="initializer_range",
        )

    @property
    def frontend_output_dim(self) -> int:
        return self.output_size
