from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for import_root in (REPOSITORY_ROOT, EXPERIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from utilis.RawBoost import process_Rawboost_feature  # noqa: E402

from rawboost_opt import Algorithm, RawBoost, RawBoostConfig  # noqa: E402


def _legacy_arguments(config: RawBoostConfig) -> SimpleNamespace:
    return SimpleNamespace(
        N_f=config.nonlinearity_order,
        nBands=config.num_bands,
        minF=config.min_frequency,
        maxF=config.max_frequency,
        minBW=config.min_bandwidth,
        maxBW=config.max_bandwidth,
        minCoeff=config.min_coefficients,
        maxCoeff=config.max_coefficients,
        minG=config.min_gain,
        maxG=config.max_gain,
        minBiasLinNonLin=config.min_nonlinear_bias,
        maxBiasLinNonLin=config.max_nonlinear_bias,
        P=config.impulse_percent,
        g_sd=config.impulse_gain,
        SNRmin=config.min_snr,
        SNRmax=config.max_snr,
    )


def _profile(function, repeats: int, sort_by: str, limit: int) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(repeats):
        function()
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(sort_by).print_stats(limit)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile legacy and optimized RawBoost algo 4.")
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "results/profile.md")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    config = RawBoostConfig()
    waveform = (
        np.random.default_rng(2026)
        .normal(
            0.0,
            0.1,
            int(config.sampling_rate * args.seconds),
        )
        .astype(np.float32)
    )
    legacy_arguments = _legacy_arguments(config)
    optimized = RawBoost(config)

    legacy = _profile(
        lambda: process_Rawboost_feature(
            waveform,
            config.sampling_rate,
            legacy_arguments,
            Algorithm.LNL_ISD_SSI.value,
        ),
        args.repeats,
        "cumulative",
        args.limit,
    )
    modern = _profile(
        lambda: optimized.augment(
            waveform,
            Algorithm.LNL_ISD_SSI,
            backend="numpy",
        ),
        args.repeats,
        "cumulative",
        args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# RawBoost Algorithm 4 Profile\n\n"
        f"Profiled {args.repeats} four-second float32 waveforms per implementation.\n\n"
        "## Legacy\n\n```text\n"
        f"{legacy}```\n\n"
        "## Optimized NumPy\n\n```text\n"
        f"{modern}```\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
