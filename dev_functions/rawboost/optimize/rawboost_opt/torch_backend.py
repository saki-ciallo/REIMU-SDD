from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .config import Algorithm, RawBoostConfig
from .parameters import RawBoostParameters


def _normalize(waveforms: torch.Tensor, *, always: bool) -> torch.Tensor:
    peak = waveforms.abs().amax(dim=-1, keepdim=True)
    denominator = peak.clamp_min(torch.finfo(waveforms.dtype).tiny)
    if always:
        return waveforms / denominator
    return torch.where(peak > 1.0, waveforms / denominator, waveforms)


def _fft_filter(
    waveforms: torch.Tensor,
    filters: Sequence[np.ndarray],
) -> torch.Tensor:
    filter_lengths = [len(coefficients) for coefficients in filters]
    lengths = torch.tensor(filter_lengths, device=waveforms.device, dtype=torch.long)
    padded = waveforms.new_zeros(len(filters), max(filter_lengths))
    for index, coefficients in enumerate(filters):
        coefficient_tensor = torch.as_tensor(
            coefficients,
            device=waveforms.device,
            dtype=waveforms.dtype,
        )
        padded[index, : coefficient_tensor.numel()] = coefficient_tensor
    convolution_length = waveforms.shape[-1] + padded.shape[-1] - 1
    fft_size = 1 << (convolution_length - 1).bit_length()
    convolved = torch.fft.irfft(
        torch.fft.rfft(waveforms, n=fft_size) * torch.fft.rfft(padded, n=fft_size),
        n=fft_size,
    )
    starts = (lengths + 1).div(2, rounding_mode="floor")
    offsets = torch.arange(waveforms.shape[-1], device=waveforms.device)
    return convolved.gather(1, starts[:, None] + offsets[None, :])


def _apply_lnl(
    waveforms: torch.Tensor,
    parameters: Sequence[RawBoostParameters],
) -> torch.Tensor:
    num_orders = {len(item.lnl.filters) for item in parameters}
    if len(num_orders) != 1:
        raise ValueError("All batch items must use the same nonlinearity order.")
    output = torch.zeros_like(waveforms)
    powered = waveforms
    for order in range(num_orders.pop()):
        if order:
            powered = powered * waveforms
        output += _fft_filter(
            powered,
            [item.lnl.filters[order] for item in parameters],
        )
    output -= output.mean(dim=-1, keepdim=True)
    return _normalize(output, always=False)


def _apply_isd(
    waveforms: torch.Tensor,
    parameters: Sequence[RawBoostParameters],
) -> torch.Tensor:
    output = waveforms.clone()
    for batch_index, item in enumerate(parameters):
        indices = torch.as_tensor(
            item.isd.indices,
            device=output.device,
            dtype=torch.long,
        )
        factors = torch.as_tensor(
            item.isd.factors,
            device=output.device,
            dtype=output.dtype,
        )
        output[batch_index, indices] += output[batch_index, indices] * factors
    return _normalize(output, always=False)


def _apply_ssi(
    waveforms: torch.Tensor,
    parameters: Sequence[RawBoostParameters],
) -> torch.Tensor:
    noise = torch.stack(
        [
            torch.as_tensor(
                item.ssi.noise,
                device=waveforms.device,
                dtype=waveforms.dtype,
            )
            for item in parameters
        ]
    )
    noise = _normalize(
        _fft_filter(noise, [item.ssi.noise_filter for item in parameters]),
        always=True,
    )
    signal_energy = waveforms.square().sum(dim=-1, keepdim=True)
    noise_energy = (
        noise.square().sum(dim=-1, keepdim=True).clamp_min(torch.finfo(waveforms.dtype).tiny)
    )
    snr = waveforms.new_tensor([item.ssi.snr for item in parameters])[:, None]
    scale = torch.sqrt(signal_energy / noise_energy) * 10.0 ** (-0.05 * snr)
    return waveforms + noise * scale


@torch.no_grad()
def augment_torch(
    waveforms: torch.Tensor,
    algorithm: Algorithm | int,
    parameters: Sequence[RawBoostParameters],
    config: RawBoostConfig,
) -> torch.Tensor:
    """Apply a batched RawBoost plan on the input tensor's current device."""

    del config  # Sampling owns config; Torch execution is always stable FP32.
    squeeze_batch = waveforms.ndim == 1
    if squeeze_batch:
        waveforms = waveforms.unsqueeze(0)
    if waveforms.ndim != 2 or waveforms.shape[-1] == 0:
        raise ValueError("waveforms must have shape [batch, samples].")
    if len(parameters) != waveforms.shape[0]:
        raise ValueError("Parameter count must match batch size.")
    algorithm = Algorithm(algorithm)
    with torch.autocast(device_type=waveforms.device.type, enabled=False):
        source = waveforms.float().contiguous()
        if algorithm is Algorithm.NONE:
            output = source.clone()
        elif algorithm is Algorithm.LNL:
            output = _apply_lnl(source, parameters)
        elif algorithm is Algorithm.ISD:
            output = _apply_isd(source, parameters)
        elif algorithm is Algorithm.SSI:
            output = _apply_ssi(source, parameters)
        elif algorithm is Algorithm.LNL_PARALLEL_ISD:
            output = _normalize(
                _apply_lnl(source, parameters) + _apply_isd(source, parameters),
                always=False,
            )
        else:
            output = source
            if algorithm in {
                Algorithm.LNL_ISD_SSI,
                Algorithm.LNL_ISD,
                Algorithm.LNL_SSI,
            }:
                output = _apply_lnl(output, parameters)
            if algorithm in {
                Algorithm.LNL_ISD_SSI,
                Algorithm.LNL_ISD,
                Algorithm.ISD_SSI,
            }:
                output = _apply_isd(output, parameters)
            if algorithm in {
                Algorithm.LNL_ISD_SSI,
                Algorithm.LNL_SSI,
                Algorithm.ISD_SSI,
            }:
                output = _apply_ssi(output, parameters)
    return output.squeeze(0) if squeeze_batch else output
