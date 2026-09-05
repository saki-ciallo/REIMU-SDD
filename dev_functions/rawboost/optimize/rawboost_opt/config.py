from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class Algorithm(IntEnum):
    NONE = 0
    LNL = 1
    ISD = 2
    SSI = 3
    LNL_ISD_SSI = 4
    LNL_ISD = 5
    LNL_SSI = 6
    ISD_SSI = 7
    LNL_PARALLEL_ISD = 8


class Backend(StrEnum):
    AUTO = "auto"
    NUMPY = "numpy"
    TORCH = "torch"


class FIRBackend(StrEnum):
    DIRECT = "direct"
    OVERLAP_ADD = "overlap_add"
    FFT = "fft"


@dataclass(frozen=True, slots=True)
class RawBoostConfig:
    """Canonical RawBoost parameters plus explicit numerical backend choices."""

    sampling_rate: int = 16_000
    num_bands: int = 5
    min_frequency: float = 20.0
    max_frequency: float = 8_000.0
    min_bandwidth: float = 100.0
    max_bandwidth: float = 1_000.0
    min_coefficients: int = 10
    max_coefficients: int = 100
    min_gain: float = 0.0
    max_gain: float = 0.0
    min_nonlinear_bias: float = 5.0
    max_nonlinear_bias: float = 20.0
    nonlinearity_order: int = 5
    impulse_percent: float = 10.0
    impulse_gain: float = 2.0
    min_snr: float = 10.0
    max_snr: float = 40.0
    fir_backend: FIRBackend | str = FIRBackend.OVERLAP_ADD
    compute_dtype: str = "float32"
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "fir_backend", FIRBackend(self.fir_backend))
        if self.compute_dtype not in {"float32", "float64"}:
            raise ValueError("compute_dtype must be 'float32' or 'float64'.")
        for name in (
            "sampling_rate",
            "num_bands",
            "min_coefficients",
            "max_coefficients",
            "nonlinearity_order",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        for name in (
            "frequency",
            "bandwidth",
            "coefficients",
            "nonlinear_bias",
            "snr",
        ):
            minimum = getattr(self, f"min_{name}")
            maximum = getattr(self, f"max_{name}")
            if minimum >= maximum:
                raise ValueError(f"min_{name} must be below max_{name}.")
        if self.min_gain > self.max_gain:
            raise ValueError("min_gain must not exceed max_gain.")
        if not 0.0 <= self.impulse_percent <= 100.0:
            raise ValueError("impulse_percent must lie in [0, 100].")
        if self.max_frequency > self.sampling_rate / 2:
            raise ValueError("max_frequency must not exceed the Nyquist frequency.")

    @property
    def numpy_dtype(self):
        import numpy as np

        return np.dtype(self.compute_dtype)
