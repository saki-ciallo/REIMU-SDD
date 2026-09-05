from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

import numpy as np
import torch

from .config import Algorithm, Backend, RawBoostConfig
from .numpy_backend import augment_numpy
from .parameters import RawBoostParameters, sample_parameters
from .torch_backend import augment_torch

type Waveform = np.ndarray | torch.Tensor | Sequence[float]


def _numpy_mono(waveform: Waveform, dtype: np.dtype) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        if waveform.is_cuda:
            raise ValueError("NumPy backend cannot consume a CUDA tensor.")
        tensor = waveform.detach()
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 1:
            raise ValueError("waveform must be mono audio shaped [samples].")
        if tensor.dtype != torch.float32 or dtype != np.dtype("float32"):
            target_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
            tensor = tensor.to(dtype=target_dtype)
        return np.ascontiguousarray(tensor.contiguous().numpy(), dtype=dtype)
    array = np.asarray(waveform, dtype=dtype)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError("waveform must be mono audio shaped [samples].")
    return np.ascontiguousarray(array)


class RawBoost:
    """Type-preserving RawBoost API with deterministic per-sample seeds."""

    def __init__(self, config: RawBoostConfig | None = None) -> None:
        self.config = config or RawBoostConfig()
        self._seed_rng = np.random.default_rng(self.config.seed)
        self._seed_lock = Lock()

    def _next_seed(self) -> int:
        with self._seed_lock:
            return int(self._seed_rng.integers(0, np.iinfo(np.int64).max))

    def parameters(
        self,
        waveform_length: int,
        algorithm: Algorithm | int,
        *,
        seed: int | None = None,
    ) -> RawBoostParameters:
        rng = np.random.default_rng(self._next_seed() if seed is None else seed)
        return sample_parameters(waveform_length, algorithm, self.config, rng)

    def augment(
        self,
        waveform: Waveform,
        algorithm: Algorithm | int,
        *,
        seed: int | None = None,
        backend: Backend | str = Backend.AUTO,
        output_type: str = "match",
    ) -> np.ndarray | torch.Tensor:
        algorithm = Algorithm(algorithm)
        backend = Backend(backend)
        if output_type not in {"match", "numpy", "torch"}:
            raise ValueError("output_type must be match, numpy, or torch.")
        source_is_tensor = isinstance(waveform, torch.Tensor)
        resolved_backend = backend
        if backend is Backend.AUTO:
            resolved_backend = (
                Backend.TORCH if source_is_tensor and waveform.is_cuda else Backend.NUMPY
            )

        if resolved_backend is Backend.NUMPY:
            source = _numpy_mono(waveform, self.config.numpy_dtype)
            parameters = self.parameters(source.size, algorithm, seed=seed)
            output = augment_numpy(source, algorithm, parameters, self.config)
        else:
            if source_is_tensor:
                tensor = waveform
            else:
                tensor = torch.from_numpy(_numpy_mono(waveform, self.config.numpy_dtype))
            parameters = self.parameters(tensor.shape[-1], algorithm, seed=seed)
            output = augment_torch(tensor, algorithm, [parameters], self.config)

        if output_type == "numpy" or (output_type == "match" and not source_is_tensor):
            if isinstance(output, torch.Tensor):
                return output.detach().cpu().numpy()
            return output
        if isinstance(output, np.ndarray):
            return torch.from_numpy(output)
        return output

    def augment_batch(
        self,
        waveforms: np.ndarray | torch.Tensor,
        algorithm: Algorithm | int,
        *,
        seeds: Sequence[int] | None = None,
        backend: Backend | str = Backend.AUTO,
    ) -> np.ndarray | torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples].")
        batch_size, waveform_length = waveforms.shape
        if seeds is not None and len(seeds) != batch_size:
            raise ValueError("seeds length must match batch size.")
        resolved_seeds = (
            list(seeds) if seeds is not None else [self._next_seed() for _ in range(batch_size)]
        )
        parameters = [
            self.parameters(waveform_length, algorithm, seed=sample_seed)
            for sample_seed in resolved_seeds
        ]
        backend = Backend(backend)
        resolved_backend = backend
        if backend is Backend.AUTO:
            resolved_backend = (
                Backend.TORCH
                if isinstance(waveforms, torch.Tensor) and waveforms.is_cuda
                else Backend.NUMPY
            )
        if resolved_backend is Backend.TORCH:
            tensor = (
                waveforms
                if isinstance(waveforms, torch.Tensor)
                else torch.from_numpy(np.asarray(waveforms, dtype=self.config.numpy_dtype))
            )
            output = augment_torch(tensor, algorithm, parameters, self.config)
            return output if isinstance(waveforms, torch.Tensor) else output.numpy()

        outputs = [
            augment_numpy(
                _numpy_mono(waveform, self.config.numpy_dtype),
                algorithm,
                item,
                self.config,
            )
            for waveform, item in zip(waveforms, parameters, strict=True)
        ]
        stacked = np.stack(outputs)
        return torch.from_numpy(stacked) if isinstance(waveforms, torch.Tensor) else stacked
