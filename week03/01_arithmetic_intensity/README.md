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

## Implementation TODOs

This lab is intentionally left as a starter exercise.

Fill in the TODO section in:

```text
arithmetic_intensity.cu
```

Required implementation work:

1. Compute the global output index.
2. Load one input value.
3. Apply a dependent `fmaf` loop `fma_repeats` times.
4. Write one output value.

The benchmark harness, timing logic, result verification, and reporting
are already provided.

The program may compile before the TODO is complete, but the benchmark
results are not meaningful until every case prints:

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
| 0 | 0.00 | | | | |
| 1 | 0.25 | | | | |
| 4 | 1.00 | | | | |
| 16 | 4.00 | | | | |
| 64 | 16.00 | | | | |
| 256 | 64.00 | | | | |
| 1024 | 256.00 | | | | |

## Detailed Timing

| FMA repeats | Min time (ms) | Max time (ms) | Avg time (ms) |
| ----------: | ------------: | ------------: | ------------: |
| 0 | | | |
| 1 | | | |
| 4 | | | |
| 16 | | | |
| 64 | | | |
| 256 | | | |
| 1024 | | | |

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


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
