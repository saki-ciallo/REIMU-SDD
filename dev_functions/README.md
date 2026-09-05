# Development Functions

This directory contains research, profiling, and implementation-comparison
code that is not required by the normal ADD training or evaluation entrypoints.

## Layout

- `dataset_loading/`: CPU and Hugging Face dataset loading benchmarks. It also
  contains the benchmark artifacts and reports produced by those experiments.
- `rawboost/cuda/`: the experimental CUDA RawBoost implementation and its
  parity/performance benchmark. The normal training path continues to use
  `utilis/RawBoost.py` through `utilis/rawboost_utils.py`.
- `rawboost/optimize/`: the independent optimized RawBoost package, including
  NumPy/Torch backends, Arrow materialization, benchmarks, and tests.
- `inference/`: Batch-1 latency and real-time-factor measurement tools.
- `analysis/`: small post-training analysis helpers such as FLOPs calculations.

Interactive model and audio debugging remains in `notebook_for_test/` because
those notebooks are already the project's established interactive test area.

## Normal runtime boundary

The root `train.py`, `test_*.py`, and `score_*.py` entrypoints do not import
the development helpers above. The runtime modules remain in `src/`, `utilis/`,
`addition_loss/`, and `configs/`. Dataset conversion and fixed-length audio
materialization remain in `datasets_process/` and `datasets_load.py`.

Experimental commands should be run from the repository root so their default
paths resolve beside the corresponding experiment.
