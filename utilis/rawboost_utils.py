from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch

from utilis.RawBoost import process_Rawboost_feature


@dataclass
class RawBoostArguments:
    """Configuration for online RawBoost applied to fixed-length waveforms."""

    rawboost_algo: int = field(
        default=0,
        metadata={
            "help": (
                "RawBoost algorithm: 0=disabled, 1=LnL, 2=ISD, 3=SSI, "
                "4=1+2+3, 5=1+2, 6=1+3, 7=2+3, 8=1||2."
            )
        },
    )
    rawboost_apply_to_validation: bool = field(
        default=True,
        metadata={"help": "Apply online RawBoost to the validation split as well as train."},
    )
    rawboost_num_bands: int = field(default=5)
    rawboost_min_frequency: float = field(default=20.0)
    rawboost_max_frequency: float = field(default=8000.0)
    rawboost_min_bandwidth: float = field(default=100.0)
    rawboost_max_bandwidth: float = field(default=1000.0)
    rawboost_min_coefficients: int = field(default=10)
    rawboost_max_coefficients: int = field(default=100)
    rawboost_min_gain: float = field(default=0.0)
    rawboost_max_gain: float = field(default=0.0)
    rawboost_min_nonlinear_bias: float = field(default=5.0)
    rawboost_max_nonlinear_bias: float = field(default=20.0)
    rawboost_nonlinearity_order: int = field(default=5)
    rawboost_impulse_percent: float = field(default=10.0)
    rawboost_impulse_gain: float = field(default=2.0)
    rawboost_min_snr: float = field(default=10.0)
    rawboost_max_snr: float = field(default=40.0)

    def __post_init__(self) -> None:
        if self.rawboost_algo not in range(9):
            raise ValueError("rawboost_algo must be an integer from 0 through 8.")
        if self.rawboost_num_bands <= 0:
            raise ValueError("rawboost_num_bands must be positive.")
        if self.rawboost_nonlinearity_order <= 0:
            raise ValueError("rawboost_nonlinearity_order must be positive.")
        if not 0.0 <= self.rawboost_impulse_percent <= 100.0:
            raise ValueError("rawboost_impulse_percent must be between 0 and 100.")
        self._validate_ordered_pair(
            "frequency",
            self.rawboost_min_frequency,
            self.rawboost_max_frequency,
        )
        self._validate_ordered_pair(
            "bandwidth",
            self.rawboost_min_bandwidth,
            self.rawboost_max_bandwidth,
        )
        self._validate_ordered_pair(
            "coefficients",
            self.rawboost_min_coefficients,
            self.rawboost_max_coefficients,
        )
        self._validate_ordered_pair(
            "gain",
            self.rawboost_min_gain,
            self.rawboost_max_gain,
            allow_equal=True,
        )
        self._validate_ordered_pair(
            "nonlinear bias",
            self.rawboost_min_nonlinear_bias,
            self.rawboost_max_nonlinear_bias,
        )
        self._validate_ordered_pair(
            "SNR",
            self.rawboost_min_snr,
            self.rawboost_max_snr,
        )

    @staticmethod
    def _validate_ordered_pair(
        name: str,
        minimum: float,
        maximum: float,
        *,
        allow_equal: bool = False,
    ) -> None:
        valid = minimum <= maximum if allow_equal else minimum < maximum
        if not valid:
            relation = "<=" if allow_equal else "<"
            raise ValueError(f"RawBoost {name} range must satisfy minimum {relation} maximum.")

    @property
    def enabled(self) -> bool:
        return self.rawboost_algo != 0

    def validate_sampling_rate(self, sampling_rate: int) -> None:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")
        if self.rawboost_max_frequency > sampling_rate / 2:
            raise ValueError(
                "rawboost_max_frequency cannot exceed the Nyquist frequency "
                f"({sampling_rate / 2:g} Hz for sampling_rate={sampling_rate})."
            )

    def to_process_args(self) -> SimpleNamespace:
        """Translate readable config names to RawBoost's original argument names."""

        return SimpleNamespace(
            nBands=self.rawboost_num_bands,
            minF=self.rawboost_min_frequency,
            maxF=self.rawboost_max_frequency,
            minBW=self.rawboost_min_bandwidth,
            maxBW=self.rawboost_max_bandwidth,
            minCoeff=self.rawboost_min_coefficients,
            maxCoeff=self.rawboost_max_coefficients,
            minG=self.rawboost_min_gain,
            maxG=self.rawboost_max_gain,
            minBiasLinNonLin=self.rawboost_min_nonlinear_bias,
            maxBiasLinNonLin=self.rawboost_max_nonlinear_bias,
            N_f=self.rawboost_nonlinearity_order,
            P=self.rawboost_impulse_percent,
            g_sd=self.rawboost_impulse_gain,
            SNRmin=self.rawboost_min_snr,
            SNRmax=self.rawboost_max_snr,
        )


class RawBoostWaveformTransform:
    """Apply RawBoost while preserving a mono waveform's length and float32 dtype."""

    def __init__(self, config: RawBoostArguments, sampling_rate: int) -> None:
        if not config.enabled:
            raise ValueError("RawBoostWaveformTransform requires rawboost_algo != 0.")
        config.validate_sampling_rate(sampling_rate)
        self.algo = config.rawboost_algo
        self.sampling_rate = sampling_rate
        self.process_args = config.to_process_args()

    def __call__(self, waveform: object) -> torch.Tensor:
        tensor = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 1:
            raise ValueError(
                "RawBoost expects a mono waveform shaped [samples] or [1, samples], "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.numel() == 0:
            raise ValueError("RawBoost cannot process an empty waveform.")

        source = tensor.contiguous().numpy()
        augmented = process_Rawboost_feature(
            source,
            self.sampling_rate,
            self.process_args,
            self.algo,
        )
        augmented_array = np.asarray(augmented, dtype=np.float32)
        if augmented_array.shape != tuple(tensor.shape):
            raise RuntimeError(
                f"RawBoost changed waveform shape from {tuple(tensor.shape)} "
                f"to {augmented_array.shape}."
            )
        if not np.isfinite(augmented_array).all():
            raise RuntimeError("RawBoost produced non-finite waveform values.")
        return torch.from_numpy(augmented_array.copy())


def build_rawboost_dataset_transform(
    waveform_transform: RawBoostWaveformTransform,
    input_values_column_name: str,
) -> Callable[[dict[str, list[object]]], dict[str, list[object]]]:
    """Build a batched Dataset.with_transform callback for fixed audio samples."""

    def transform(batch: dict[str, list[object]]) -> dict[str, list[object]]:
        batch[input_values_column_name] = [
            waveform_transform(waveform)
            for waveform in batch[input_values_column_name]
        ]
        return batch

    return transform


__all__ = [
    "RawBoostArguments",
    "RawBoostWaveformTransform",
    "build_rawboost_dataset_transform",
]
