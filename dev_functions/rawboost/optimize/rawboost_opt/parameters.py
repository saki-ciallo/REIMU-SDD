from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import Algorithm, RawBoostConfig


@dataclass(frozen=True, slots=True)
class LnLParameters:
    filters: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class ISDParameters:
    indices: np.ndarray
    factors: np.ndarray


@dataclass(frozen=True, slots=True)
class SSIParameters:
    noise: np.ndarray
    noise_filter: np.ndarray
    snr: float


@dataclass(frozen=True, slots=True)
class RawBoostParameters:
    lnl: LnLParameters | None = None
    isd: ISDParameters | None = None
    ssi: SSIParameters | None = None


def _uniform_interval(
    rng: np.random.Generator,
    start: float,
    stop: float,
    size: int | None = None,
) -> float | np.ndarray:
    """Sample along start -> stop, including legacy descending intervals."""

    return start + (stop - start) * rng.random(size)


def generate_notch_coefficients(
    config: RawBoostConfig,
    rng: np.random.Generator,
    *,
    min_gain: float,
    max_gain: float,
) -> np.ndarray:
    coefficients = np.ones(1, dtype=np.float64)
    for _ in range(config.num_bands):
        center = _uniform_interval(rng, config.min_frequency, config.max_frequency)
        bandwidth = _uniform_interval(rng, config.min_bandwidth, config.max_bandwidth)
        order = int(_uniform_interval(rng, config.min_coefficients, config.max_coefficients))
        if order % 2 == 0:
            order += 1
        low = max(center - bandwidth / 2, 1e-3)
        high = min(center + bandwidth / 2, config.sampling_rate / 2 - 1e-3)
        bandstop = signal.firwin(
            order,
            [float(low), float(high)],
            window="hamming",
            fs=config.sampling_rate,
        )
        coefficients = np.convolve(coefficients, bandstop)

    gain = _uniform_interval(rng, min_gain, max_gain)
    _, response = signal.freqz(coefficients, 1, fs=config.sampling_rate)
    peak = max(float(np.abs(response).max()), np.finfo(np.float64).tiny)
    coefficients = 10.0 ** (gain / 20.0) * coefficients / peak
    return np.asarray(coefficients, dtype=config.numpy_dtype)


def _sample_lnl(config: RawBoostConfig, rng: np.random.Generator) -> LnLParameters:
    filters: list[np.ndarray] = []
    min_gain = config.min_gain
    max_gain = config.max_gain
    for order in range(config.nonlinearity_order):
        if order == 1:
            min_gain -= config.min_nonlinear_bias
            max_gain -= config.max_nonlinear_bias
        filters.append(
            generate_notch_coefficients(
                config,
                rng,
                min_gain=min_gain,
                max_gain=max_gain,
            )
        )
    return LnLParameters(tuple(filters))


def _sample_isd(
    waveform_length: int,
    config: RawBoostConfig,
    rng: np.random.Generator,
) -> ISDParameters:
    ratio = _uniform_interval(rng, 0.0, config.impulse_percent) / 100.0
    num_impulses = int(waveform_length * ratio)
    indices = rng.choice(waveform_length, size=num_impulses, replace=False)
    factors = config.impulse_gain * (
        _uniform_interval(rng, -1.0, 1.0, num_impulses)
        * _uniform_interval(rng, -1.0, 1.0, num_impulses)
    )
    return ISDParameters(
        indices=np.asarray(indices, dtype=np.int64),
        factors=np.asarray(factors, dtype=config.numpy_dtype),
    )


def _sample_ssi(
    waveform_length: int,
    config: RawBoostConfig,
    rng: np.random.Generator,
) -> SSIParameters:
    noise = rng.standard_normal(waveform_length).astype(config.numpy_dtype, copy=False)
    return SSIParameters(
        noise=noise,
        noise_filter=generate_notch_coefficients(
            config,
            rng,
            min_gain=config.min_gain,
            max_gain=config.max_gain,
        ),
        snr=float(_uniform_interval(rng, config.min_snr, config.max_snr)),
    )


def sample_parameters(
    waveform_length: int,
    algorithm: Algorithm | int,
    config: RawBoostConfig,
    rng: np.random.Generator,
) -> RawBoostParameters:
    """Sample only the primitive parameter sets required by one algorithm."""

    if waveform_length <= 0:
        raise ValueError("waveform_length must be positive.")
    algorithm = Algorithm(algorithm)
    needs_lnl = algorithm in {
        Algorithm.LNL,
        Algorithm.LNL_ISD_SSI,
        Algorithm.LNL_ISD,
        Algorithm.LNL_SSI,
        Algorithm.LNL_PARALLEL_ISD,
    }
    needs_isd = algorithm in {
        Algorithm.ISD,
        Algorithm.LNL_ISD_SSI,
        Algorithm.LNL_ISD,
        Algorithm.ISD_SSI,
        Algorithm.LNL_PARALLEL_ISD,
    }
    needs_ssi = algorithm in {
        Algorithm.SSI,
        Algorithm.LNL_ISD_SSI,
        Algorithm.LNL_SSI,
        Algorithm.ISD_SSI,
    }
    return RawBoostParameters(
        lnl=_sample_lnl(config, rng) if needs_lnl else None,
        isd=_sample_isd(waveform_length, config, rng) if needs_isd else None,
        ssi=_sample_ssi(waveform_length, config, rng) if needs_ssi else None,
    )
