# RawBoost CPU Scaling

Algorithm 4 on four-second, 16 kHz float32 waveforms. Resources means NumPy worker processes or Torch intra-op threads.

| implementation | resources | batch | ms/sample | samples/s |
|---|---:|---:|---:|---:|
| numpy_process | 1 | 64 | 4.215 | 237.2 |
| numpy_process | 2 | 64 | 2.692 | 371.4 |
| numpy_process | 4 | 64 | 1.397 | 715.9 |
| numpy_process | 8 | 64 | 0.938 | 1066.1 |
| numpy_process | 16 | 64 | 0.591 | 1692.1 |
| numpy_process | 24 | 64 | 0.706 | 1416.8 |
| numpy_process | 32 | 64 | 0.717 | 1394.5 |
| torch_cpu_end_to_end | 1 | 64 | 6.005 | 166.5 |
| torch_cpu_execution_only | 1 | 64 | 3.812 | 262.3 |
| torch_cpu_end_to_end | 2 | 64 | 4.618 | 216.5 |
| torch_cpu_execution_only | 2 | 64 | 2.380 | 420.1 |
| torch_cpu_end_to_end | 4 | 64 | 3.876 | 258.0 |
| torch_cpu_execution_only | 4 | 64 | 1.663 | 601.3 |
| torch_cpu_end_to_end | 8 | 64 | 3.679 | 271.8 |
| torch_cpu_execution_only | 8 | 64 | 1.371 | 729.5 |
| torch_cpu_end_to_end | 16 | 64 | 3.540 | 282.5 |
| torch_cpu_execution_only | 16 | 64 | 1.200 | 833.6 |
| torch_cpu_end_to_end | 24 | 64 | 3.691 | 270.9 |
| torch_cpu_execution_only | 24 | 64 | 1.432 | 698.3 |
| torch_cpu_end_to_end | 32 | 64 | 3.669 | 272.5 |
| torch_cpu_execution_only | 32 | 64 | 1.333 | 750.1 |
