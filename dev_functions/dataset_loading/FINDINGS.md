# Dataset loading findings

## Decision

Keep the current fixed Hugging Face Arrow dataset loaded with
`load_from_disk()` for normal local training.

It is the fastest drop-in path under the current training settings
(`batch_size=32`, `dataloader_num_workers=2`) and retains map-style random
access and ordinary epoch shuffling. Converting the fixed waveforms to
Parquet does not improve this path:

- non-streaming `load_dataset("parquet")` first materializes an Arrow cache,
  so repeated training reads Arrow again;
- direct Parquet streaming needs substantially more workers to reach peak
  throughput and does not provide the same global random-access shuffle;
- raw FLAC saves disk space but pays decoding and resizing cost on every epoch.

This agrees with the Hugging Face guidance that local Arrow is uncompressed
and faster to reload, while Parquet is better suited to compact distribution,
upload, download, and querying:

- https://huggingface.co/docs/datasets/en/process#save
- https://huggingface.co/docs/datasets/en/loading#arrow
- https://huggingface.co/docs/datasets/en/loading#parquet

## Test setup

- CPU: AMD Ryzen 9 9950X, 16 cores / 32 threads
- Filesystem: local ext4
- Python: 3.12.13
- PyTorch: 2.12.0+cu130, CPU tensors only
- datasets: 5.0.0
- pyarrow: 24.0.0
- Source: 4,096 randomly selected ASVspoof2019 train utterances
- Logical sample: float32 waveform with 64,000 values
- Parquet: 8 shards, Snappy and Zstd tested independently
- Main matrix: batch sizes 8 and 32; workers 0, 2, 4, and 8
- Confirmation run: batch size 32, workers 0, 2, and 8, five repetitions

The benchmark uses the same collator behavior as training and reports the
median across repetitions. GPU transfer and model compute are deliberately
excluded. Repeated measurements benefit from the operating system page cache,
so absolute values are specific to this machine; the relative behavior and
startup/throughput tradeoffs are the useful comparison.

## Output equivalence

All six paths were compared on 64 samples against the Arrow reference:

| Path | Labels | Shapes | Waveforms | Max abs error | Max RMSE |
|---|---|---|---|---:|---:|
| Arrow | exact | exact | exact | 0 | 0 |
| Parquet cached, Snappy | exact | exact | exact | 0 | 0 |
| Parquet cached, Zstd | exact | exact | exact | 0 | 0 |
| Parquet streaming, Snappy | exact | exact | exact | 0 | 0 |
| Parquet streaming, Zstd | exact | exact | exact | 0 | 0 |
| Raw FLAC | exact | exact | exact | 0 | 0 |

Evidence:
`results/dataset-validation-20260731-023517.json`.

## Current training configuration

The current common config uses batch size 32 and two DataLoader workers.
The five-repeat confirmation result is:

| Path | First batch | Samples/s | p95 batch wait |
|---|---:|---:|---:|
| Arrow | 0.013 s | **11,420.1** | 3.38 ms |
| Parquet cached, Snappy | 0.013 s | 11,126.5 | 3.60 ms |
| Parquet cached, Zstd | 0.014 s | 11,062.6 | 3.53 ms |
| Parquet streaming, Zstd | 0.326 s | 3,573.1 | 7.20 ms |
| Parquet streaming, Snappy | 0.351 s | 3,316.0 | 7.07 ms |
| Raw FLAC | 0.036 s | 2,847.3 | 21.69 ms |

At the actual trainer settings, Arrow is approximately:

- 4.01 times the raw FLAC throughput;
- 3.20 times the direct Zstd Parquet streaming throughput;
- 3.44 times the direct Snappy Parquet streaming throughput.

The cached Parquet variants are within a few percent of Arrow because they
are reading materialized Arrow cache files, not compressed Parquet in the
steady-state loop.

Evidence:
`results/dataset-loading-20260731-023655.{json,csv,md}`.

## Maximum measured throughput

The result changes if the goal is strictly maximum sequential CPU throughput
and eight workers are available:

| Path, batch 32 | Workers | Samples/s | First batch |
|---|---:|---:|---:|
| Parquet streaming, Zstd | 8 | **16,101.4** | 0.395 s |
| Parquet streaming, Snappy | 8 | 15,972.9 | 0.441 s |
| Parquet cached, Zstd | 8 | 14,674.2 | 0.026 s |
| Arrow | 8 | 14,570.1 | 0.022 s |
| Raw FLAC | 8 | 10,044.6 | 0.059 s |

This Parquet streaming peak is not a drop-in replacement for the current
training behavior. The benchmark subset was randomized at preparation time
and only Parquet shard order was shuffled at load time. A streaming
`IterableDataset` normally uses buffered rather than exact global shuffling.
Hugging Face also assigns iterable shards across DataLoader workers, so this
result depends on having at least as many useful shards as workers:

- https://huggingface.co/docs/datasets/en/stream#shuffle
- https://huggingface.co/docs/datasets/en/stream#convert-from-a-dataset

With zero workers, Arrow was the fastest direct local path at 14,975.6
samples/s. This isolated result includes no GPU compute to overlap with data
loading, so it does not by itself prove that `dataloader_num_workers=0` is
best for full training.

## Disk and startup tradeoffs

Equivalent 4,096-sample artifacts occupied:

| Storage | Size |
|---|---:|
| Arrow | 1,000.1 MiB |
| Parquet, Snappy | 799.4 MiB |
| Parquet, Zstd | 424.8 MiB |
| Selected raw FLAC | 244.9 MiB |
| Raw path manifest | 0.4 MiB |

Each non-streaming Parquet variant additionally created a 1,000.1 MiB Arrow
cache. Keeping both the compressed source and generated cache therefore uses
more local disk than keeping Arrow alone.

For the complete ASVspoof2019 train split:

- fixed Arrow: 6,498,771,842 bytes;
- raw FLAC directory: 1,586,320,281 bytes;
- raw FLAC is about 4.10 times smaller.

A cold `load_dataset("audiofolder", data_dir=..., split="train")` call scanned
all three directory splits and took 43.98 seconds before returning the 25,380
training rows on this machine. The raw benchmark intentionally uses a
prebuilt path-only Arrow manifest, making its approximately 4 ms dataset
startup a favorable estimate; it still remains much slower during iteration
because decoding occurs per sample. Hugging Face documents that audio is
decoded while iterating and that local fast disks do not always benefit from
additional decoder threads:

- https://huggingface.co/docs/datasets/en/audio_load#audio-decoding

## Practical recommendation

1. Continue training from `ASVspoof2019_16k_4s_fixed` with
   `load_from_disk()`.
2. Do not convert the local fixed training dataset to Parquet merely to improve
   training speed. Non-streaming loading converts it back to Arrow cache.
3. Keep two workers as a conservative default. Test four workers in a complete
   GPU training profile if DataLoader waits are visible; the dataset-only
   matrix reached 12,204.7 samples/s with four workers.
4. Use Zstd Parquet streaming only when reduced persistent storage is more
   important than exact map-style shuffle and eight CPU workers are acceptable.
5. Use raw FLAC only when the roughly 4.1-fold disk reduction is worth recurring
   decode cost. Keep a prebuilt manifest rather than invoking `audiofolder`
   discovery at every run.

The fixed Arrow path already supplies more than eleven thousand examples per
second in the current CPU-only test. If the real SSL training loop processes
far fewer examples per second, dataset storage is unlikely to be the active
bottleneck; an end-to-end Trainer profile is the correct next measurement
before spending more complexity on the input path.
