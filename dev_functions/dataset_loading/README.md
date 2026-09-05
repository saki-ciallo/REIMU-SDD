# Dataset loading experiment

This directory isolates CPU-only benchmarks for the ADD audio input pipeline.
It does not modify the training code.

The consolidated conclusion is in [FINDINGS.md](FINDINGS.md).

## Compared paths

- `arrow`: local `save_to_disk` / `load_from_disk` Arrow with memory mapping.
- `parquet_cached_*`: Parquet loaded through `load_dataset`, which materializes
  a reusable Arrow cache before map-style training.
- `parquet_streaming_*`: direct sequential Parquet reads without materializing
  the full dataset. Samples are randomized when the benchmark subset is
  prepared, and Parquet shard order is shuffled at load time. This preserves
  parallel worker sharding under HF Datasets 5.0.
- `raw_audio`: an Arrow manifest of FLAC paths decoded on access through the
  Hugging Face `Audio` feature and TorchCodec.

All variants use the same selected utterances and return the same
`float32 [batch, 64000]` tensors to an equivalent copy of the current training
collator. Generated multi-gigabyte artifacts are ignored by Git.

## Smoke test

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py312

python dev_functions/dataset_loading/benchmark_dataset_loading.py all \
  --sample-count 256 \
  --variants arrow,parquet_cached_snappy,parquet_streaming_snappy,raw_audio \
  --batch-sizes 8,32 \
  --num-workers 0,2 \
  --warmup-batches 2 \
  --measured-batches 4 \
  --repeats 1
```

## Main benchmark

```bash
python dev_functions/dataset_loading/benchmark_dataset_loading.py all \
  --sample-count 4096 \
  --batch-sizes 8,32 \
  --num-workers 0,2,4,8 \
  --warmup-batches 8 \
  --measured-batches 64 \
  --repeats 2
```

Preparation and benchmarking can also be run independently:

```bash
python dev_functions/dataset_loading/benchmark_dataset_loading.py prepare \
  --sample-count 4096

python dev_functions/dataset_loading/benchmark_dataset_loading.py validate \
  --sample-count 4096 \
  --validation-samples 64

python dev_functions/dataset_loading/benchmark_dataset_loading.py benchmark \
  --sample-count 4096
```

`validate` compares every selected storage path against Arrow without
shuffling. It checks waveform shapes, labels, exact float32 values, maximum
absolute error, and RMSE.

The report separates:

- dataset object load and cache-preparation time;
- first-batch latency, including worker startup and initial prefetch;
- steady-state samples per second;
- logical waveform MiB per second;
- median and p95 batch wait time.

Results are written to `dev_functions/dataset_loading/results`.

The default benchmark enables `persistent_workers` when workers are present
and uses `prefetch_factor=2`. Each benchmark case creates a fresh DataLoader,
so persistent workers do not conceal worker startup in the first-batch metric.
Use `--no-persistent-workers` to mirror a configuration that restarts workers
between epochs.
