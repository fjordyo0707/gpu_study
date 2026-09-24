# Experiment 03 - Memory-Bound vs Compute-Bound Workloads

## Objective

Build two controlled kernels that make the difference between
memory-bound and compute-bound behavior obvious.

The arithmetic-intensity lab changed one knob: the amount of math per byte.
This lab turns that idea into a direct comparison between workload shapes:

- a streaming memory kernel
- a dependent compute kernel
- a mixed kernel that sits between the two

## First Question

Can we predict which kernels are memory-bound or compute-bound from their
arithmetic intensity and measured performance?

## Hypothesis

The streaming kernel should be limited mostly by global-memory bandwidth
because it performs very little math per byte.

The dependent compute kernel should become compute-bound as `fma_repeats`
increases because it reads and writes only one float per element but performs
many dependent FMA operations.

The mixed kernel should sit between those two extremes.

## Recommended Reading

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  review effective bandwidth, arithmetic throughput, and the APOD
  optimization workflow.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html):
  read the memory hierarchy and arithmetic instruction throughput sections
  as background for why different kernels hit different limits.
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785):
  the model this lab is preparing you to use more confidently.
- [8 Steps to 3.7 TFLOP/s on NVIDIA V100 GPU: Roofline Analysis and Other Tricks](https://arxiv.org/abs/2008.11326):
  readable case study showing how optimization choices move a kernel
  between bottlenecks.

## Implementation TODOs

This lab is intentionally left as a starter exercise.

Fill in the TODO sections in:

```text
memory_vs_compute.cu
```

Required implementation work:

1. Implement `memory_stream_kernel`.
2. Implement `compute_chain_kernel`.
3. Implement `mixed_kernel`.
4. Keep each kernel's memory traffic matching the comments in the code.
5. Do not change the timing harness until all cases print `PASS`.

The benchmark harness, timing logic, FLOP/byte calculation, and verification
logic are already provided.

## Build

```bash
make
```

Equivalent manual command:

```bash
nvcc -O3 -std=c++17 -arch=sm_61 memory_vs_compute.cu -o memory_vs_compute
```

## Run

Default:

```bash
./memory_vs_compute
```

Arguments:

```bash
./memory_vs_compute <elements> <threads_per_block> <iterations>
```

Example:

```bash
./memory_vs_compute 16777216 256 100
```

## Workloads

The lab compares three shapes:

| Workload | Useful memory traffic | FLOPs per element | Expected behavior |
| -------- | --------------------: | ----------------: | ----------------- |
| memory_stream | 16 bytes | 4 | Memory-bound |
| compute_chain_64 | 8 bytes | 128 | Transition region |
| compute_chain_512 | 8 bytes | 1024 | Compute-bound |
| mixed_64 | 12 bytes | 129 | Middle case |

## Suggested Measurements

| Workload | Arithmetic intensity (FLOP/byte) | Avg time (ms) | GFLOP/s | Effective bandwidth (GB/s) | Result |
| -------- | -------------------------------: | ------------: | ------: | -------------------------: | ------ |
| memory_stream | | | | | |
| compute_chain_64 | | | | | |
| compute_chain_512 | | | | | |
| mixed_64 | | | | | |

## Detailed Timing

| Workload | Min time (ms) | Max time (ms) | Avg time (ms) |
| -------- | ------------: | ------------: | ------------: |
| memory_stream | | | |
| compute_chain_64 | | | |
| compute_chain_512 | | | |
| mixed_64 | | | |

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


## Connection To Previous Labs

The arithmetic-intensity sweep showed that runtime starts changing once
enough math is added per byte.

This lab gives those points names:

- memory-bound: performance follows bytes moved
- compute-bound: performance follows operations executed
- mixed: both memory and arithmetic matter

## Next Experiment

Use the measurements from this lab and the Roofline lab to practice
bottleneck diagnosis:

```text
../04_bottleneck_analysis
```
