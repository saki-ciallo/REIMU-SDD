# Diagnostics

Diagnostics inspect a constructed model or a completed run. They do not add
layers to the forward path and are not required for ordinary inference.

## Files

```text
diagnostics/
├── __init__.py      public diagnostics API
├── model_report.py  component tree and parameter counts
├── run_manifest.py  resolved config, versions, GPU, and startup timings
└── benchmark.py     Batch-1 latency and real-time factor measurement
```

## Usage flow

```text
ADDModel + ResolvedExperiment
              |
              +--> model_report.py
              |       -> model_overview.md
              |
              +--> run_manifest.py
              |       -> run_manifest.json
              |
              +--> benchmark.py (optional saved checkpoint measurement)
                      -> inference_benchmark.json
```

`model_report.py` distinguishes frontend, backbone architecture, mixer,
feed-forward, pooling, and classifier components and records total/trainable
parameter counts. `run_manifest.py` records the exact resolved configuration
and runtime metadata so two experiments can be compared. `benchmark.py`
measures warmed-up CUDA inference latency and derives the real-time factor;
checkpoint loading and warmup are excluded from the measured interval.
