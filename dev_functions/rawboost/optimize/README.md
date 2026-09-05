# RawBoost Optimize

This directory is an isolated optimization and validation workspace for
`utilis/RawBoost.py`. The legacy implementation is imported only by the benchmark and profiler;
it is not modified or used by the optimized package.

## Scope

The optimized implementation keeps the original RawBoost recipes and default parameters while
making data types, random state, execution backend, and persistence explicit.

| Algo | Recipe |
|---:|---|
| 0 | disabled |
| 1 | linear and nonlinear convolutive noise (LnL) |
| 2 | impulsive signal-dependent noise (ISD) |
| 3 | stationary signal-independent noise (SSI) |
| 4 | LnL -> ISD -> SSI |
| 5 | LnL -> ISD |
| 6 | LnL -> SSI |
| 7 | ISD -> SSI |
| 8 | normalized LnL + ISD parallel sum |

The canonical defaults live in `rawboost_opt/config.py` and match the active values in
`configs/train_common.yaml`: 16 kHz, five bands, nonlinearity order five, impulse gain two,
10 percent maximum impulses, and 10-40 dB SSI SNR.

## Existing Training Data Flow

The current main project loads fixed four-second waveforms from a Hugging Face Arrow
`DatasetDict`.

For algorithms other than 4, `train.py` attaches a `Dataset.with_transform` callback. The
current legacy callback performs this path for every sample:

```text
Arrow list/torch tensor
  -> CPU float32 torch tensor
  -> contiguous NumPy view
  -> legacy RawBoost (often promotes to float64)
  -> NumPy float32 conversion and copy
  -> new CPU torch tensor
  -> DataLoader collator
```

Algorithm 4 follows a different path: collation happens first, the batch is moved to CUDA, and
`ADDTrainer` applies `ExperimentalCudaRawBoostAlgo4` immediately before the model. Validation is
augmented only when `rawboost_apply_to_validation` is enabled.

The repeated per-sample conversions are not the only bottleneck. Profiling shows that the legacy
LnL stage dominates because it performs five polynomial branches and many direct FIR filters.

## Optimized Design

The package separates random parameter sampling from waveform execution:

```text
RawBoostConfig
  -> sample_parameters(seed) -> RawBoostParameters
  -> NumPy overlap-add/direct OR batched Torch FFT
  -> output with the original shape and float32 dtype
```

- `rawboost_opt/api.py`: type-preserving single and batched public API.
- `rawboost_opt/parameters.py`: deterministic parameter plans shared by CPU and CUDA.
- `rawboost_opt/numpy_backend.py`: float32 NumPy implementation with overlap-add FIR by default.
- `rawboost_opt/torch_backend.py`: batched Torch FFT implementation on CPU or CUDA.
- `rawboost_opt/parallel.py`: optional persistent thread/process executors for offline work.
- `rawboost_opt/huggingface.py`: deterministic `Dataset.map` and Arrow persistence.
- `benchmark_cpu_scaling.py`: NumPy process and Torch CPU thread scaling benchmark.

The legacy descending nonlinear gain interval is intentionally retained. The old global
`np.random` stream and the new `Generator` do not produce byte-identical random plans for the same
seed, but each distribution, algorithm order, and default range is preserved. Supplying the same
`RawBoostParameters` gives NumPy direct, overlap-add, Torch CPU, and Torch CUDA results within the
tested floating-point tolerance.

## Type and Copy Policy

Use float32 at every data boundary.

| Context | Recommended input | Backend | Recommended output |
|---|---|---|---|
| Offline Arrow materialization | Arrow `List(float32)` | NumPy overlap-add | Arrow `List(float32)` |
| Online DataLoader worker | contiguous CPU torch float32 | NumPy overlap-add | CPU torch float32 |
| Online after collation | CUDA torch `[B, samples]` | Torch FFT | CUDA torch float32 |
| Numerical reference | NumPy float32/float64 | NumPy direct | matching NumPy dtype |

A contiguous CPU float32 tensor is exposed to NumPy as a shared-memory view. The augmented result
is new storage, and `torch.from_numpy` can share that output without another copy. A CUDA tensor
never travels through NumPy. The Hugging Face mapper requests NumPy formatting directly so Arrow
does not first decode each waveform into a Python list.

## Hugging Face Arrow

Offline materialization removes augmentation from the training hot path and is the preferred
choice when a fixed augmented corpus is acceptable:

```bash
conda activate py312
python dev_functions/rawboost/optimize/build_arrow_dataset.py \
  ../datasets/ASVspoof2019_16k_4s_fixed \
  ../datasets/ASVspoof2019_16k_4s_rawboost4 \
  --algorithm 4 \
  --splits train validation \
  --num-proc 8 \
  --batch-size 32
```

Each sample receives a seed derived from `(base_seed, split_name, row_index)`. Results therefore
remain the same across `num_proc` values and processing order. The output directory includes
`rawboost_manifest.json` with the source, parameters, split sizes, and elapsed time. The builder
refuses to overwrite an existing directory.

Run Arrow multiprocessing from a standalone process before starting CUDA or other threaded
runtimes. Hugging Face `datasets` currently uses a fork-based multiprocessing implementation on
Linux; nesting it inside DataLoader workers or another process pool is not recommended.

## Online API

Single waveform:

```python
from rawboost_opt import Algorithm, RawBoost, RawBoostConfig

augmenter = RawBoost(RawBoostConfig(seed=42))
waveform = augmenter.augment(waveform, Algorithm.LNL_ISD_SSI, seed=sample_index)
```

CUDA batch after collation:

```python
seeds = [base_seed + index for index in sample_indices]
input_values = augmenter.augment_batch(
    input_values.cuda(),
    Algorithm.LNL_ISD_SSI,
    seeds=seeds,
    backend="torch",
)
```

For CPU offline batches, `ParallelBatchProcessor(..., kind="process")` keeps workers alive across
batches. Do not create this executor inside a DataLoader worker.

## Performance Findings

The checked-in benchmark was run on 4-second, 16 kHz float32 waveforms with a 32-core host and an
RTX 4080 SUPER. See `results/benchmark.md` and `results/profile.md` for the full tables.

Key algorithm 4 results:

| Path | Median time/sample | Throughput |
|---|---:|---:|
| Legacy NumPy direct | 22.419 ms | 44.6 samples/s |
| Optimized NumPy overlap-add | 5.955 ms | 167.9 samples/s |
| Optimized NumPy direct | 12.009 ms | 83.3 samples/s |
| Optimized CPU process pool, 8 workers, batch 16 | 1.185 ms | 843.8 samples/s |
| Optimized CUDA FFT, batch 16 | 3.824 ms | 261.5 samples/s |

The legacy profiler attributes 0.418 of 0.503 seconds to LnL over 20 calls. Direct FIR filtering
accounts for 0.185 seconds and filter generation for 0.102 seconds. In the optimized profile,
parameter sampling takes 0.093 of 0.152 seconds and waveform application takes 0.058 seconds;
`firwin` and overlap-add FFT are the remaining leading operations.

Threads are a poor default here: two to eight threads were slower than serial execution because
SciPy FFT/filter work already uses native resources and concurrent calls contend for them.
Processes scaled well through eight workers in this test. CUDA is useful when augmentation must
remain online and the batch is already on-device, but it does not beat eight CPU processes for
offline preprocessing because random FIR plans are still sampled on the CPU.

### CPU Scaling

The local Ryzen 9 9950X exposes 16 physical cores and 32 logical CPUs. A batch-64 scaling run
measured the complete algorithm 4 path, including random filter generation:

| Backend | Resources | Samples/s |
|---|---:|---:|
| NumPy serial | 1 process | 237.2 |
| NumPy multiprocessing | 8 processes | 1066.1 |
| NumPy multiprocessing | 16 processes | 1692.1 |
| NumPy multiprocessing | 24 processes | 1416.8 |
| NumPy multiprocessing | 32 processes | 1394.5 |
| Torch CPU end-to-end | 8 threads | 271.8 |
| Torch CPU end-to-end | 16 threads | 282.5 |
| Torch CPU execution only | 16 threads | 833.6 |

A second batch-128, five-repeat run confirmed the plateau: NumPy produced 1674.0 samples/s with
16 processes, 1700.5 with 24, and 1580.7 with 32. The 1.6 percent difference between 16 and 24 is
small enough to treat as workload/scheduler variation; 16 processes is the robust offline default.
When training concurrently on the same host, start with eight workers so data augmentation does
not consume every physical core needed by decoding, collation, logging, and the training process.

Torch CPU is not the preferred complete backend here. Its batch FFT execution is faster than
serial NumPy, but FIR parameter plans are generated sequentially with SciPy before tensor
execution. NumPy multiprocessing parallelizes both plan generation and filtering. Increasing
Torch from 8 to 16-32 threads therefore gives little end-to-end benefit and can increase memory
bandwidth and FFT scheduling contention. Full results are in `results/cpu_scaling.md` and
`results/cpu_scaling_confirm/cpu_scaling.md`.

## Reproduce

```bash
conda activate py312
PYTHONPATH=dev_functions/rawboost/optimize pytest -q dev_functions/rawboost/optimize/tests
python dev_functions/rawboost/optimize/benchmark.py --repeats 3 --batch-size 16
python dev_functions/rawboost/optimize/benchmark_cpu_scaling.py --batch-size 64 --repeats 3
python dev_functions/rawboost/optimize/profile_algo4.py
```

The package is loaded through `PYTHONPATH` for these local experiments, so an
editable install and its generated `build`/`egg-info` metadata are not needed.

The test suite covers algorithms 0-8, deterministic seeds, zero waveforms, float32 preservation,
direct versus overlap-add parity, NumPy versus Torch CPU/CUDA parity, thread/process execution,
and deterministic Arrow output across process counts.

## Recommended Integration Order

1. Use offline Arrow materialization for fixed augmentation experiments.
2. If every epoch needs fresh noise, use NumPy overlap-add in existing DataLoader workers without
   an additional executor.
3. Use the batched Torch backend only after collation and device transfer.
4. Keep `FIRBackend.DIRECT` as a reference/debug mode, not the training default.
5. Benchmark on the deployment server before selecting process count; eight workers is a local
   result, not a universal constant.
