# Experiment 02 - Cache Behavior

## Objective

Measure how repeated reads from different working-set sizes change useful
memory throughput.

The memory-coalescing lab showed that neighboring threads need neighboring
addresses for efficient global-memory transactions. This lab asks the next
question: after the access pattern is reasonable, how much does cache
reuse help?

## First Question

When each thread repeatedly reads from the same working set, what happens
as the working set grows from cache-friendly to streaming-size?

## Hypothesis

Small working sets should achieve high useful bandwidth because repeated
reads can be served from cache.

Large working sets should eventually become slower because the data no
longer fits well in cache, so more reads must be served from DRAM.

## Recommended Reading

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html):
  use the memory hierarchy and L2 cache control material as the reference
  model for this lab. Some advanced cache-control APIs may target newer
  GPUs than the GTX 1080 Ti, but the concepts still matter.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  read the device memory spaces, L2 cache, and shared-memory sections to
  separate cache reuse from explicit shared-memory reuse.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html):
  the essential follow-up when you want cache hit rates and memory
  workload counters instead of only timing numbers.
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/):
  useful contrast reading because shared memory is programmer-managed
  cache-like storage, unlike hardware-managed L1/L2 cache.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135):
  a modern and readable paper about reducing traffic between GPU memory
  levels with tiling and reuse.
- [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/abs/2507.10789):
  latest stretch paper; read the cache, memory hierarchy, and scheduling
  sections for research-style examples of what to measure next.

## Implementation TODOs

This lab is intentionally left as a starter exercise.

Fill in the TODO section in:

```text
cache_behavior.cu
```

Required implementation work:

1. Compute the global output index.
2. Map each thread to a base index inside the active working set.
3. Repeatedly read from the working set using the documented index formula.
4. Accumulate the values and write the final sum to output.

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
nvcc -O3 -std=c++17 -arch=sm_61 cache_behavior.cu -o cache_behavior
```

## Run

Default:

```bash
./cache_behavior
```

Arguments:

```bash
./cache_behavior <output_elements> <threads_per_block> <repeat_reads> <iterations>
```

Examples:

```bash
./cache_behavior 1048576 256 64 100
./cache_behavior 2097152 256 64 100
```

Make shortcut:

```bash
make run
```

## Access Pattern

Each output element is a sum of repeated reads from a bounded working set:

```text
base        = output_index & (working_set_elements - 1)
input_index = (base + repeat_index * 131) & (working_set_elements - 1)
output[i]   = sum(input[input_index])
```

The working-set sizes are powers of two so the mask operation wraps the
index back into the active working set.

## Suggested Measurements

| Working set | Working-set elements | Repeat reads/thread | Avg time (ms) | Useful bandwidth (GB/s) | Result |
| ----------- | -------------------: | ------------------: | ------------: | ----------------------: | ------ |
| 4 KiB       |                 1024 |                  64 |               |                         |        |
| 64 KiB      |                16384 |                  64 |               |                         |        |
| 1 MiB       |               262144 |                  64 |               |                         |        |
| 8 MiB       |              2097152 |                  64 |               |                         |        |
| 64 MiB      |             16777216 |                  64 |               |                         |        |
| 256 MiB     |             67108864 |                  64 |               |                         |        |

## Detailed Timing

| Working set | Min time (ms) | Max time (ms) | Avg time (ms) |
| ----------- | ------------: | ------------: | ------------: |
| 4 KiB       |               |               |               |
| 64 KiB      |               |               |               |
| 1 MiB       |               |               |               |
| 8 MiB       |               |               |               |
| 64 MiB      |               |               |               |
| 256 MiB     |               |               |               |

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


## Connection To Previous Labs

The memory-coalescing lab controlled the shape of the warp's global-memory
addresses. This cache-behavior lab keeps the access formula regular and
changes the amount of data being reused.

Together, the two labs separate two different memory questions:

- coalescing: are neighboring threads asking for compact addresses?
- caching: are repeated reads hitting data that is still nearby in cache?

## Next Experiment

After cache behavior, continue deeper into Week 2 memory hierarchy:

- shared-memory access patterns
- bank conflicts
- global memory versus shared memory reuse
