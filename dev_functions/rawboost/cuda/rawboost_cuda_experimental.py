from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from utilis.RawBoost import filterFIR, genNotchCoeffs, normWav, randRange
from utilis.rawboost_utils import RawBoostArguments


@dataclass(frozen=True)
class RawBoostAlgo4Parameters:
    """Random state required to replay one RawBoost algo4 augmentation."""

    lnl_filters: tuple[np.ndarray, ...]
    isd_indices: np.ndarray
    isd_factors: np.ndarray
    ssi_noise: np.ndarray
    ssi_filter: np.ndarray
    ssi_snr: float


def sample_rawboost_algo4_parameters(
    waveform_length: int,
    args: SimpleNamespace,
    sampling_rate: int = 16_000,
) -> RawBoostAlgo4Parameters:
    """Sample algo4 parameters in the same order as the original NumPy code."""

    if waveform_length <= 0:
        raise ValueError("waveform_length must be positive.")

    min_gain = args.minG
    max_gain = args.maxG
    lnl_filters = []
    for order in range(args.N_f):
        if order == 1:
            min_gain -= args.minBiasLinNonLin
            max_gain -= args.maxBiasLinNonLin
        lnl_filters.append(
            genNotchCoeffs(
                args.nBands,
                args.minF,
                args.maxF,
                args.minBW,
                args.maxBW,
                args.minCoeff,
                args.maxCoeff,
                min_gain,
                max_gain,
                sampling_rate,
            )
        )

    beta = randRange(0, args.P, 0)
    num_impulses = int(waveform_length * (beta / 100))
    isd_indices = np.random.permutation(waveform_length)[:num_impulses]
    isd_factors = args.g_sd * (
        (2 * np.random.rand(num_impulses) - 1)
        * (2 * np.random.rand(num_impulses) - 1)
    )

    ssi_noise = np.random.normal(0, 1, waveform_length)
    ssi_filter = genNotchCoeffs(
        args.nBands,
        args.minF,
        args.maxF,
        args.minBW,
        args.maxBW,
        args.minCoeff,
        args.maxCoeff,
        args.minG,
        args.maxG,
        sampling_rate,
    )
    ssi_snr = randRange(args.SNRmin, args.SNRmax, 0)
    return RawBoostAlgo4Parameters(
        lnl_filters=tuple(lnl_filters),
        isd_indices=isd_indices,
        isd_factors=isd_factors,
        ssi_noise=ssi_noise,
        ssi_filter=ssi_filter,
        ssi_snr=ssi_snr,
    )


def apply_rawboost_algo4_numpy(
    waveform: np.ndarray,
    parameters: RawBoostAlgo4Parameters,
) -> np.ndarray:
    """Replay algo4 with the original SciPy filtering operations."""

    waveform = np.asarray(waveform)
    if waveform.ndim != 1:
        raise ValueError("waveform must have shape [samples].")
    if len(parameters.lnl_filters) == 0:
        raise ValueError("parameters must contain at least one LnL filter.")

    output = np.zeros(waveform.shape[0], dtype=np.float64)
    for order, coefficients in enumerate(parameters.lnl_filters, start=1):
        output += filterFIR(np.power(waveform, order), coefficients)
    output -= np.mean(output)
    output = normWav(output, 0)

    output = output.copy()
    indices = parameters.isd_indices
    output[indices] += output[indices] * parameters.isd_factors
    output = normWav(output, 0)

    noise = filterFIR(parameters.ssi_noise, parameters.ssi_filter)
    noise = normWav(noise, 1)
    noise *= (
        np.sqrt(np.vdot(output, output) / np.vdot(noise, noise))
        * 10 ** (-0.05 * parameters.ssi_snr)
    )
    return output + noise


def _normalize_waveform(waveforms: torch.Tensor, *, always: bool) -> torch.Tensor:
    peak = waveforms.abs().amax(dim=-1, keepdim=True)
    if always:
        return waveforms / peak
    return torch.where(peak > 1, waveforms / peak, waveforms)


def _fft_fir_filter(
    waveforms: torch.Tensor,
    filters: Sequence[np.ndarray],
) -> torch.Tensor:
    """Apply one distinct FIR to every row using batched linear convolution."""

    if waveforms.ndim != 2:
        raise ValueError("waveforms must have shape [batch_size, num_samples].")
    if len(filters) != waveforms.shape[0]:
        raise ValueError("The number of filters must match batch_size.")

    device = waveforms.device
    dtype = waveforms.dtype
    lengths = torch.tensor(
        [len(coefficients) for coefficients in filters],
        device=device,
        dtype=torch.long,
    )
    max_filter_length = int(lengths.max().item())
    padded_filters = torch.zeros(
        (len(filters), max_filter_length),
        device=device,
        dtype=dtype,
    )
    for index, coefficients in enumerate(filters):
        coefficient_tensor = torch.as_tensor(
            np.asarray(coefficients),
            device=device,
            dtype=dtype,
        )
        padded_filters[index, : coefficient_tensor.numel()] = coefficient_tensor

    convolution_length = waveforms.shape[-1] + max_filter_length - 1
    fft_size = 1 << (convolution_length - 1).bit_length()
    waveform_spectrum = torch.fft.rfft(waveforms, n=fft_size)
    filter_spectrum = torch.fft.rfft(padded_filters, n=fft_size)
    convolved = torch.fft.irfft(
        waveform_spectrum * filter_spectrum,
        n=fft_size,
    )

    starts = torch.div(lengths + 1, 2, rounding_mode="floor")
    sample_offsets = torch.arange(
        waveforms.shape[-1],
        device=device,
        dtype=torch.long,
    )
    gather_indices = starts[:, None] + sample_offsets[None, :]
    return convolved.gather(dim=1, index=gather_indices)


def _apply_rawboost_algo4_cuda_impl(
    waveforms: torch.Tensor,
    parameters: Sequence[RawBoostAlgo4Parameters],
) -> torch.Tensor:
    squeeze_batch = waveforms.ndim == 1
    if squeeze_batch:
        waveforms = waveforms.unsqueeze(0)
    if waveforms.ndim != 2:
        raise ValueError("waveforms must have shape [batch_size, num_samples].")
    if not waveforms.is_cuda:
        raise ValueError("apply_rawboost_algo4_cuda requires CUDA waveforms.")
    if len(parameters) != waveforms.shape[0]:
        raise ValueError("The number of parameter sets must match batch_size.")

    waveforms = waveforms.float()
    num_orders = {len(item.lnl_filters) for item in parameters}
    if len(num_orders) != 1:
        raise ValueError("All batch items must use the same nonlinear order.")
    nonlinear_order = num_orders.pop()
    if nonlinear_order == 0:
        raise ValueError("At least one nonlinear order is required.")

    output = torch.zeros_like(waveforms)
    for order in range(1, nonlinear_order + 1):
        filters = [item.lnl_filters[order - 1] for item in parameters]
        output += _fft_fir_filter(waveforms.pow(order), filters)
    output -= output.mean(dim=-1, keepdim=True)
    output = _normalize_waveform(output, always=False)

    impulse_factors = torch.zeros_like(output)
    for batch_index, item in enumerate(parameters):
        indices = torch.as_tensor(
            item.isd_indices,
            device=output.device,
            dtype=torch.long,
        )
        factors = torch.as_tensor(
            item.isd_factors,
            device=output.device,
            dtype=output.dtype,
        )
        impulse_factors[batch_index, indices] = factors
    output = output + output * impulse_factors
    output = _normalize_waveform(output, always=False)

    noise = torch.stack(
        [
            torch.as_tensor(
                item.ssi_noise,
                device=output.device,
                dtype=output.dtype,
            )
            for item in parameters
        ]
    )
    noise = _fft_fir_filter(
        noise,
        [item.ssi_filter for item in parameters],
    )
    noise = _normalize_waveform(noise, always=True)
    signal_energy = output.square().sum(dim=-1, keepdim=True)
    noise_energy = noise.square().sum(dim=-1, keepdim=True)
    snr = torch.tensor(
        [item.ssi_snr for item in parameters],
        device=output.device,
        dtype=output.dtype,
    ).unsqueeze(-1)
    noise *= torch.sqrt(signal_energy / noise_energy) * torch.pow(
        output.new_tensor(10.0),
        -0.05 * snr,
    )
    output = output + noise
    return output.squeeze(0) if squeeze_batch else output


@torch.no_grad()
def apply_rawboost_algo4_cuda(
    waveforms: torch.Tensor,
    parameters: Sequence[RawBoostAlgo4Parameters],
) -> torch.Tensor:
    """Apply experimental batched CUDA algo4 using FP32 FFT FIR filtering."""

    with torch.autocast(device_type="cuda", enabled=False):
        return _apply_rawboost_algo4_cuda_impl(waveforms, parameters)


class ExperimentalCudaRawBoostAlgo4(nn.Module):
    """Experimental CUDA algo4 bound to the canonical RawBoost configuration."""

    def __init__(
        self,
        config: RawBoostArguments | None = None,
        sampling_rate: int = 16_000,
    ) -> None:
        super().__init__()
        self.config = (
            RawBoostArguments(rawboost_algo=4)
            if config is None
            else config
        )
        if self.config.rawboost_algo != 4:
            raise ValueError(
                "ExperimentalCudaRawBoostAlgo4 requires rawboost_algo=4."
            )
        self.config.validate_sampling_rate(sampling_rate)
        self.sampling_rate = sampling_rate
        self.process_args = self.config.to_process_args()

    def sample_parameters(
        self,
        batch_size: int,
        waveform_length: int,
    ) -> tuple[RawBoostAlgo4Parameters, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return tuple(
            sample_rawboost_algo4_parameters(
                waveform_length,
                self.process_args,
                sampling_rate=self.sampling_rate,
            )
            for _ in range(batch_size)
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        parameters: Sequence[RawBoostAlgo4Parameters] | None = None,
    ) -> torch.Tensor:
        batch_size = 1 if waveforms.ndim == 1 else waveforms.shape[0]
        if parameters is None:
            parameters = self.sample_parameters(
                batch_size=batch_size,
                waveform_length=waveforms.shape[-1],
            )
        return apply_rawboost_algo4_cuda(waveforms, parameters)


__all__ = [
    "ExperimentalCudaRawBoostAlgo4",
    "RawBoostAlgo4Parameters",
    "apply_rawboost_algo4_cuda",
    "apply_rawboost_algo4_numpy",
    "sample_rawboost_algo4_parameters",
]
