# RawBoost Benchmark

All values use 16 kHz, four-second float32 waveforms. Lower ms/sample is better.

## Algorithm 0-8, Single Sample CPU

| implementation | algo | batch | workers | ms/sample | samples/s |
|---|---:|---:|---:|---:|---:|
| legacy_numpy_direct | 0 | 1 | 1 | 0.001 | 1490313.6 |
| optimized_numpy_overlap_add | 0 | 1 | 1 | 0.013 | 79529.2 |
| optimized_numpy_direct | 0 | 1 | 1 | 0.009 | 117288.3 |
| legacy_numpy_direct | 1 | 1 | 1 | 14.850 | 67.3 |
| optimized_numpy_overlap_add | 1 | 1 | 1 | 3.010 | 332.2 |
| optimized_numpy_direct | 1 | 1 | 1 | 7.293 | 137.1 |
| legacy_numpy_direct | 2 | 1 | 1 | 0.531 | 1884.2 |
| optimized_numpy_overlap_add | 2 | 1 | 1 | 0.041 | 24161.0 |
| optimized_numpy_direct | 2 | 1 | 1 | 0.036 | 27802.5 |
| legacy_numpy_direct | 3 | 1 | 1 | 2.626 | 380.9 |
| optimized_numpy_overlap_add | 3 | 1 | 1 | 1.778 | 562.6 |
| optimized_numpy_direct | 3 | 1 | 1 | 2.656 | 376.5 |
| legacy_numpy_direct | 4 | 1 | 1 | 22.419 | 44.6 |
| optimized_numpy_overlap_add | 4 | 1 | 1 | 5.955 | 167.9 |
| optimized_numpy_direct | 4 | 1 | 1 | 12.009 | 83.3 |
| legacy_numpy_direct | 5 | 1 | 1 | 19.735 | 50.7 |
| optimized_numpy_overlap_add | 5 | 1 | 1 | 2.932 | 341.1 |
| optimized_numpy_direct | 5 | 1 | 1 | 6.730 | 148.6 |
| legacy_numpy_direct | 6 | 1 | 1 | 20.801 | 48.1 |
| optimized_numpy_overlap_add | 6 | 1 | 1 | 5.364 | 186.4 |
| optimized_numpy_direct | 6 | 1 | 1 | 12.038 | 83.1 |
| legacy_numpy_direct | 7 | 1 | 1 | 3.682 | 271.6 |
| optimized_numpy_overlap_add | 7 | 1 | 1 | 1.678 | 596.1 |
| optimized_numpy_direct | 7 | 1 | 1 | 2.373 | 421.4 |
| legacy_numpy_direct | 8 | 1 | 1 | 17.828 | 56.1 |
| optimized_numpy_overlap_add | 8 | 1 | 1 | 3.055 | 327.4 |
| optimized_numpy_direct | 8 | 1 | 1 | 6.931 | 144.3 |

## Algorithm 4 Components

| implementation | algo | batch | workers | ms/sample | samples/s |
|---|---:|---:|---:|---:|---:|
| parameter_sampling | 4 | 1 | 1 | 2.159 | 463.2 |
| waveform_application | 4 | 1 | 1 | 1.881 | 531.6 |

## Algorithm 4 CPU Parallelism

| implementation | algo | batch | workers | ms/sample | samples/s |
|---|---:|---:|---:|---:|---:|
| serial | 4 | 16 | 1 | 4.209 | 237.6 |
| thread | 4 | 16 | 2 | 8.932 | 112.0 |
| thread | 4 | 16 | 4 | 12.155 | 82.3 |
| thread | 4 | 16 | 8 | 12.625 | 79.2 |
| process | 4 | 16 | 2 | 2.643 | 378.3 |
| process | 4 | 16 | 4 | 1.498 | 667.7 |
| process | 4 | 16 | 8 | 1.185 | 843.8 |

## Algorithm 0-8, CUDA Batch

| implementation | algo | batch | workers | ms/sample | samples/s |
|---|---:|---:|---:|---:|---:|
| optimized_torch_fft | 0 | 16 | 1 | 0.009 | 108770.3 |
| optimized_torch_fft | 1 | 16 | 1 | 1.650 | 606.2 |
| optimized_torch_fft | 2 | 16 | 1 | 0.173 | 5768.2 |
| optimized_torch_fft | 3 | 16 | 1 | 1.058 | 945.1 |
| optimized_torch_fft | 4 | 16 | 1 | 3.824 | 261.5 |
| optimized_torch_fft | 5 | 16 | 1 | 1.948 | 513.5 |
| optimized_torch_fft | 6 | 16 | 1 | 2.597 | 385.1 |
| optimized_torch_fft | 7 | 16 | 1 | 1.065 | 939.3 |
| optimized_torch_fft | 8 | 16 | 1 | 1.751 | 571.2 |
