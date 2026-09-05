# RawBoost CPU Scaling

Algorithm 4 on four-second, 16 kHz float32 waveforms. Resources means NumPy worker processes or Torch intra-op threads.

| implementation | resources | batch | ms/sample | samples/s |
|---|---:|---:|---:|---:|
| numpy_process | 8 | 128 | 0.873 | 1145.3 |
| numpy_process | 16 | 128 | 0.597 | 1674.0 |
| numpy_process | 24 | 128 | 0.588 | 1700.5 |
| numpy_process | 32 | 128 | 0.633 | 1580.7 |
| torch_cpu_end_to_end | 8 | 128 | 3.779 | 264.6 |
| torch_cpu_execution_only | 8 | 128 | 1.547 | 646.4 |
| torch_cpu_end_to_end | 16 | 128 | 3.889 | 257.1 |
| torch_cpu_execution_only | 16 | 128 | 1.666 | 600.4 |
| torch_cpu_end_to_end | 24 | 128 | 3.914 | 255.5 |
| torch_cpu_execution_only | 24 | 128 | 1.753 | 570.4 |
| torch_cpu_end_to_end | 32 | 128 | 3.902 | 256.3 |
| torch_cpu_execution_only | 32 | 128 | 1.705 | 586.5 |
