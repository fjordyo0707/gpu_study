# Experiment 01 - Arithmetic Intensity

## Objective

Measure how increasing arithmetic work per byte changes kernel
performance.

Week 2 focused on memory hierarchy. Week 3 starts performance modeling:
before using Roofline, you need to measure arithmetic intensity.

## First Question

At what point does a kernel stop behaving like a memory-bandwidth test and
start behaving more like a compute-throughput test?

## Hypothesis

When each element performs very little math, performance should be limited
mostly by memory traffic.

As each element performs more dependent floating-point work while reading
and writing the same amount of memory, arithmetic intensity increases.
Eventually runtime should scale more with FLOP count than with bytes moved.

## Recommended Reading

- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785):
  the classic paper that defines the performance-model idea used in this
  week.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  review effective bandwidth, FLOP counting, and profiling workflow.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html):
  use this later to compare your calculated GFLOP/s with profiler
  counters.
- [Hierarchical Roofline Analysis: How to Collect Data using Performance Tools on Intel CPUs and NVIDIA GPUs](https://arxiv.org/abs/2009.02449):
  practical next step after you understand simple arithmetic intensity.
- [8 Steps to 3.7 TFLOP/s on NVIDIA V100 GPU: Roofline Analysis and Other Tricks](https://arxiv.org/abs/2008.11326):
  readable case study showing how Roofline thinking guides real
  optimization.

## Implementation Status

This lab has been completed and recorded in:

```text
std_record.log
```

The critical implementation work was in:

```text
arithmetic_intensity.cu
```

Implemented kernel tasks:

1. Compute the global output index.
2. Load one input value.
3. Apply a dependent `fmaf` loop `fma_repeats` times.
4. Write one output value.

The benchmark harness, timing logic, result verification, and reporting
were provided.

The benchmark result is valid because every measured case printed:

```text
Result              = PASS
```

## Build

```bash
make
```

Equivalent manual command:

```bash
nvcc -O3 -std=c++17 -arch=sm_61 arithmetic_intensity.cu -o arithmetic_intensity
```

## Run

Default:

```bash
./arithmetic_intensity
```

Arguments:

```bash
./arithmetic_intensity <elements> <threads_per_block> <iterations>
```

Example:

```bash
./arithmetic_intensity 16777216 256 100
```

## Metrics

Each element reads one float and writes one float:

```text
bytes per element = 8
```

Each `fmaf` is counted as two FLOPs:

```text
FLOPs per element = 2 * fma_repeats
arithmetic intensity = FLOPs / bytes
```

## Suggested Measurements

| FMA repeats | Arithmetic intensity (FLOP/byte) | Avg time (ms) | GFLOP/s | Effective bandwidth (GB/s) | Result |
| ----------: | -------------------------------: | ------------: | ------: | -------------------------: | ------ |
| 0 | 0.00 | 0.373 | 0.000 | 359.566 | PASS |
| 1 | 0.25 | 0.373 | 89.853 | 359.413 | PASS |
| 4 | 1.00 | 0.374 | 359.160 | 359.160 | PASS |
| 16 | 4.00 | 0.374 | 1436.489 | 359.122 | PASS |
| 64 | 16.00 | 0.470 | 4568.001 | 285.500 | PASS |
| 256 | 64.00 | 1.282 | 6698.538 | 104.665 | PASS |
| 1024 | 256.00 | 4.490 | 7652.792 | 29.894 | PASS |

Recorded from:

```text
./arithmetic_intensity 16777216 256 100
```

## Detailed Timing

| FMA repeats | Min time (ms) | Max time (ms) | Avg time (ms) |
| ----------: | ------------: | ------------: | ------------: |
| 0 | 0.372 | 0.385 | 0.373 |
| 1 | 0.371 | 0.381 | 0.373 |
| 4 | 0.371 | 0.390 | 0.374 |
| 16 | 0.372 | 0.382 | 0.374 |
| 64 | 0.454 | 0.536 | 0.470 |
| 256 | 1.264 | 1.431 | 1.282 |
| 1024 | 4.203 | 5.155 | 4.490 |

## Roofline Inputs From This Lab

These numbers are useful for the next lab:

| Quantity | Value | Note |
| -------- | ----: | ---- |
| Best measured low-AI bandwidth | 359.566 GB/s | From the `0` FMA case |
| Best measured throughput in this kernel | 7652.792 GFLOP/s | From the `1024` FMA case |
| Approximate measured ridge point | 21.284 FLOP/byte | `7652.792 / 359.566` |

The measured ridge point is only for this benchmark. The dependent FMA loop
may not reach the GPU's theoretical FP32 peak because every operation
depends on the previous value.

## Observation Questions

1.
2.
3.

## Observation

The first four cases had almost identical runtime:

- `0` repeats: 0.373 ms
- `1` repeat: 0.373 ms
- `4` repeats: 0.374 ms
- `16` repeats: 0.374 ms

That means the extra arithmetic up to `16` dependent FMA operations per
element was mostly hidden by the fixed memory traffic and launch/runtime
costs. The effective bandwidth stayed near 359 GB/s for all of those cases.

The transition started around `64` repeats. Runtime increased to 0.470 ms,
GFLOP/s jumped to 4568.001, and effective bandwidth dropped to 285.500
GB/s. By `256` and `1024` repeats, runtime scaled much more clearly with
the amount of arithmetic.

The highest measured throughput was 7652.792 GFLOP/s at `1024` repeats.
That is the best compute-side number from this specific dependent-FMA
microbenchmark, not necessarily the GPU's absolute FP32 peak.

## Interpretation

The hypothesis is supported.

Low arithmetic-intensity cases behave like bandwidth tests. They move the
same 128 MiB of useful data and finish in about the same time even when a
small amount of arithmetic is added.

Higher arithmetic-intensity cases behave more like compute tests. The bytes
moved do not change, but the runtime grows because each thread now performs
many dependent FMA operations before writing its result.

The most important lesson is that "effective bandwidth" becomes less
meaningful once the kernel is compute-bound. The high-FMA cases report low
bandwidth not because DRAM became slower, but because the kernel spends most
of its time doing arithmetic instead of moving data.

For the next Roofline lab, use the low-AI bandwidth plateau as the measured
memory roof and the high-AI throughput as a measured compute roof for this
controlled benchmark. The observed transition near 16 to 64 FLOP/byte is
consistent with the calculated measured ridge point of about 21.284
FLOP/byte.

## Connection To Previous Labs

Vector addition was low arithmetic intensity.

Matrix multiplication had much higher arithmetic intensity.

This lab creates a controlled bridge between those two extremes by
changing only the amount of arithmetic per element.

## Next Experiment

Use these measurements to build a simple Roofline model:

```text
../02_roofline_model
```
