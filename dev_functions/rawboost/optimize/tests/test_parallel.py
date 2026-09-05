from __future__ import annotations

import numpy as np
import pytest

from rawboost_opt import Algorithm, RawBoostConfig
from rawboost_opt.parallel import ParallelBatchProcessor, ParallelConfig


@pytest.mark.parametrize(
    ("kind", "workers"),
    [("serial", 1), ("thread", 2), ("process", 2)],
)
def test_parallel_processor_is_deterministic(kind: str, workers: int) -> None:
    rng = np.random.default_rng(9)
    waveforms = rng.normal(0.0, 0.1, (4, 2_000)).astype(np.float32)
    seeds = [10, 11, 12, 13]
    config = RawBoostConfig(seed=1)

    with ParallelBatchProcessor(
        config,
        Algorithm.LNL_ISD,
        ParallelConfig(kind=kind, workers=workers),
    ) as processor:
        output = processor.process(waveforms, seeds=seeds)

    with ParallelBatchProcessor(config, Algorithm.LNL_ISD) as serial:
        expected = serial.process(waveforms, seeds=seeds)

    assert output.shape == waveforms.shape
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, expected)
