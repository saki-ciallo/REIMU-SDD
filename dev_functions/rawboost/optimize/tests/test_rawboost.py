from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from rawboost_opt import Algorithm, FIRBackend, RawBoost, RawBoostConfig
from rawboost_opt.numpy_backend import augment_numpy
from rawboost_opt.torch_backend import augment_torch


@pytest.fixture
def waveform() -> np.ndarray:
    rng = np.random.default_rng(2026)
    return rng.normal(0.0, 0.15, 8_000).astype(np.float32)


@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_numpy_algorithms_are_deterministic_and_float32(
    waveform: np.ndarray,
    algorithm: Algorithm,
) -> None:
    augmenter = RawBoost(RawBoostConfig(seed=17))
    first = augmenter.augment(waveform, algorithm, seed=91, backend="numpy")
    second = augmenter.augment(waveform, algorithm, seed=91, backend="numpy")

    assert isinstance(first, np.ndarray)
    assert first.shape == waveform.shape
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_overlap_add_matches_direct_for_identical_parameters(
    waveform: np.ndarray,
    algorithm: Algorithm,
) -> None:
    direct_config = RawBoostConfig(fir_backend=FIRBackend.DIRECT, seed=12)
    overlap_config = replace(direct_config, fir_backend=FIRBackend.OVERLAP_ADD)
    augmenter = RawBoost(direct_config)
    parameters = augmenter.parameters(len(waveform), algorithm, seed=77)

    direct = augment_numpy(waveform, algorithm, parameters, direct_config)
    overlap = augment_numpy(waveform, algorithm, parameters, overlap_config)

    np.testing.assert_allclose(overlap, direct, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_torch_matches_numpy_for_identical_parameters(
    waveform: np.ndarray,
    algorithm: Algorithm,
) -> None:
    config = RawBoostConfig(fir_backend=FIRBackend.OVERLAP_ADD, seed=4)
    augmenter = RawBoost(config)
    parameters = augmenter.parameters(len(waveform), algorithm, seed=123)
    expected = augment_numpy(waveform, algorithm, parameters, config)
    actual = augment_torch(
        torch.from_numpy(waveform),
        algorithm,
        [parameters],
        config,
    ).numpy()

    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_cuda_batch_matches_numpy(waveform: np.ndarray, algorithm: Algorithm) -> None:
    config = RawBoostConfig(seed=29)
    augmenter = RawBoost(config)
    seeds = [101, 202]
    batch = np.stack([waveform, waveform[::-1].copy()])
    expected = augmenter.augment_batch(batch, algorithm, seeds=seeds, backend="numpy")
    actual = augmenter.augment_batch(
        torch.from_numpy(batch).cuda(),
        algorithm,
        seeds=seeds,
        backend="torch",
    )

    assert isinstance(actual, torch.Tensor)
    assert actual.is_cuda
    assert actual.dtype == torch.float32
    np.testing.assert_allclose(
        actual.cpu().numpy(),
        expected,
        rtol=5e-4,
        atol=5e-5,
    )


def test_zero_waveform_stays_finite() -> None:
    waveform = np.zeros(2_000, dtype=np.float32)
    augmenter = RawBoost(RawBoostConfig())

    for algorithm in Algorithm:
        output = augmenter.augment(waveform, algorithm, seed=algorithm.value)
        assert np.isfinite(output).all()
        np.testing.assert_array_equal(output, np.zeros_like(output))


def test_type_preserving_api(waveform: np.ndarray) -> None:
    augmenter = RawBoost(RawBoostConfig())
    numpy_output = augmenter.augment(waveform, Algorithm.ISD, seed=3)
    torch_output = augmenter.augment(torch.from_numpy(waveform), Algorithm.ISD, seed=3)

    assert isinstance(numpy_output, np.ndarray)
    assert isinstance(torch_output, torch.Tensor)
    np.testing.assert_array_equal(torch_output.numpy(), numpy_output)
