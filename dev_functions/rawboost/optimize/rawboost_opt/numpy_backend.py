from __future__ import annotations

import numpy as np
from scipy import signal

from .config import Algorithm, FIRBackend, RawBoostConfig
from .parameters import ISDParameters, LnLParameters, RawBoostParameters, SSIParameters


def normalize_waveform(waveform: np.ndarray, *, always: bool) -> np.ndarray:
    peak = float(np.abs(waveform).max(initial=0.0))
    if peak == 0.0 or (not always and peak <= 1.0):
        return waveform
    return waveform / peak


def filter_fir(
    waveform: np.ndarray,
    coefficients: np.ndarray,
    backend: FIRBackend,
) -> np.ndarray:
    """Apply the original centered FIR crop with a selectable convolution engine."""

    num_samples = waveform.shape[-1]
    start = (coefficients.shape[0] + 1) // 2
    if backend is FIRBackend.DIRECT:
        padding = coefficients.shape[0] + 1
        padded = np.pad(waveform, (0, padding))
        filtered = signal.lfilter(coefficients, 1, padded)
        return filtered[start : start + num_samples]
    if backend is FIRBackend.OVERLAP_ADD:
        filtered = signal.oaconvolve(waveform, coefficients, mode="full")
    else:
        filtered = signal.fftconvolve(waveform, coefficients, mode="full")
    return filtered[start : start + num_samples]


def apply_lnl(
    waveform: np.ndarray,
    parameters: LnLParameters,
    config: RawBoostConfig,
) -> np.ndarray:
    output = np.zeros_like(waveform, dtype=config.numpy_dtype)
    powered = np.asarray(waveform, dtype=config.numpy_dtype)
    for order, coefficients in enumerate(parameters.filters):
        if order:
            powered = powered * waveform
        output += filter_fir(powered, coefficients, config.fir_backend)
    output -= output.mean(dtype=np.float64)
    return normalize_waveform(output, always=False)


def apply_isd(
    waveform: np.ndarray,
    parameters: ISDParameters,
) -> np.ndarray:
    output = waveform.copy()
    indices = parameters.indices
    output[indices] += output[indices] * parameters.factors
    return normalize_waveform(output, always=False)


def apply_ssi(
    waveform: np.ndarray,
    parameters: SSIParameters,
    config: RawBoostConfig,
) -> np.ndarray:
    noise = filter_fir(
        parameters.noise,
        parameters.noise_filter,
        config.fir_backend,
    )
    noise = normalize_waveform(noise, always=True)
    signal_energy = float(np.vdot(waveform, waveform).real)
    noise_energy = max(
        float(np.vdot(noise, noise).real),
        np.finfo(config.numpy_dtype).tiny,
    )
    scale = np.sqrt(signal_energy / noise_energy) * 10.0 ** (-0.05 * parameters.snr)
    return np.asarray(waveform + noise * scale, dtype=config.numpy_dtype)


def augment_numpy(
    waveform: np.ndarray,
    algorithm: Algorithm | int,
    parameters: RawBoostParameters,
    config: RawBoostConfig,
) -> np.ndarray:
    """Apply one RawBoost plan and always return contiguous float32/float64 NumPy."""

    algorithm = Algorithm(algorithm)
    source = np.asarray(waveform, dtype=config.numpy_dtype)
    if source.ndim == 2 and source.shape[0] == 1:
        source = source[0]
    if source.ndim != 1 or source.size == 0:
        raise ValueError("waveform must be non-empty mono audio shaped [samples].")
    source = np.ascontiguousarray(source)
    if algorithm is Algorithm.NONE:
        return source.copy()
    if algorithm is Algorithm.LNL:
        return np.ascontiguousarray(
            apply_lnl(source, parameters.lnl, config),
            dtype=config.numpy_dtype,
        )
    if algorithm is Algorithm.ISD:
        return np.ascontiguousarray(
            apply_isd(source, parameters.isd),
            dtype=config.numpy_dtype,
        )
    if algorithm is Algorithm.SSI:
        return np.ascontiguousarray(
            apply_ssi(source, parameters.ssi, config),
            dtype=config.numpy_dtype,
        )
    if algorithm is Algorithm.LNL_PARALLEL_ISD:
        output = apply_lnl(source, parameters.lnl, config) + apply_isd(
            source,
            parameters.isd,
        )
        return np.ascontiguousarray(
            normalize_waveform(output, always=False),
            dtype=config.numpy_dtype,
        )

    output = source
    if algorithm in {Algorithm.LNL_ISD_SSI, Algorithm.LNL_ISD, Algorithm.LNL_SSI}:
        output = apply_lnl(output, parameters.lnl, config)
    if algorithm in {Algorithm.LNL_ISD_SSI, Algorithm.LNL_ISD, Algorithm.ISD_SSI}:
        output = apply_isd(output, parameters.isd)
    if algorithm in {Algorithm.LNL_ISD_SSI, Algorithm.LNL_SSI, Algorithm.ISD_SSI}:
        output = apply_ssi(output, parameters.ssi, config)
    return np.ascontiguousarray(output, dtype=config.numpy_dtype)
