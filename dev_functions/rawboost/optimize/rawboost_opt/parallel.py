from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .api import RawBoost
from .config import Algorithm, RawBoostConfig

type ExecutorKind = Literal["serial", "thread", "process"]

_PROCESS_AUGMENTER: RawBoost | None = None
_PROCESS_ALGORITHM: Algorithm | None = None


def _initialize_process(config: RawBoostConfig, algorithm: Algorithm) -> None:
    global _PROCESS_AUGMENTER, _PROCESS_ALGORITHM
    _PROCESS_AUGMENTER = RawBoost(config)
    _PROCESS_ALGORITHM = algorithm


def _process_one(payload: tuple[np.ndarray, int]) -> np.ndarray:
    waveform, seed = payload
    if _PROCESS_AUGMENTER is None or _PROCESS_ALGORITHM is None:
        raise RuntimeError("RawBoost process worker was not initialized.")
    return _PROCESS_AUGMENTER.augment(
        waveform,
        _PROCESS_ALGORITHM,
        seed=seed,
        backend="numpy",
        output_type="numpy",
    )


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    kind: ExecutorKind = "serial"
    workers: int = 1
    process_start_method: str = "spawn"

    def __post_init__(self) -> None:
        if self.kind not in {"serial", "thread", "process"}:
            raise ValueError("kind must be serial, thread, or process.")
        if self.workers <= 0:
            raise ValueError("workers must be positive.")
        if self.kind == "serial" and self.workers != 1:
            raise ValueError("serial execution requires workers=1.")
        if self.process_start_method not in mp.get_all_start_methods():
            raise ValueError(f"Unsupported process_start_method={self.process_start_method!r}.")


class ParallelBatchProcessor:
    """Persistent optional executor for offline batches outside DataLoader workers."""

    def __init__(
        self,
        rawboost_config: RawBoostConfig,
        algorithm: Algorithm | int,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        self.rawboost_config = rawboost_config
        self.algorithm = Algorithm(algorithm)
        self.parallel_config = parallel_config or ParallelConfig()
        self.augmenter = RawBoost(rawboost_config)
        self.executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None
        if self.parallel_config.kind == "thread":
            self.executor = ThreadPoolExecutor(
                max_workers=self.parallel_config.workers,
                thread_name_prefix="rawboost",
            )
        elif self.parallel_config.kind == "process":
            context = mp.get_context(self.parallel_config.process_start_method)
            self.executor = ProcessPoolExecutor(
                max_workers=self.parallel_config.workers,
                mp_context=context,
                initializer=_initialize_process,
                initargs=(rawboost_config, self.algorithm),
            )

    def _thread_one(self, payload: tuple[np.ndarray, int]) -> np.ndarray:
        waveform, seed = payload
        return self.augmenter.augment(
            waveform,
            self.algorithm,
            seed=seed,
            backend="numpy",
            output_type="numpy",
        )

    def process(
        self,
        waveforms: np.ndarray,
        *,
        seeds: list[int] | None = None,
    ) -> np.ndarray:
        waveforms = np.asarray(waveforms, dtype=self.rawboost_config.numpy_dtype)
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples].")
        if seeds is None:
            seeds = [self.rawboost_config.seed + index for index in range(len(waveforms))]
        if len(seeds) != len(waveforms):
            raise ValueError("seeds length must match batch size.")
        payloads = list(zip(waveforms, seeds, strict=True))
        if self.executor is None:
            outputs = map(self._thread_one, payloads)
        elif self.parallel_config.kind == "thread":
            outputs = self.executor.map(self._thread_one, payloads)
        else:
            outputs = self.executor.map(_process_one, payloads, chunksize=1)
        return np.stack(list(outputs))

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def __enter__(self) -> ParallelBatchProcessor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
