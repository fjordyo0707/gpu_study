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

## Implementation Status

This lab has been completed and recorded in:

```text
std_record.log
```

The critical implementation work was in:

```text
cache_behavior.cu
```

Implemented kernel tasks:

1. Compute the global output index.
2. Map each thread to a base index inside the active working set.
3. Repeatedly read from the working set using the documented index formula.
4. Accumulate the values and write the final sum to output.

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

## Recorded Runs

Run 1:

```text
./cache_behavior 1048576 256 64 100
```

| Setting | Value |
| ------- | ----: |
| Output elements | 1048576 |
| Largest input | 256.000 MiB |
| Output bytes | 4.000 MiB |
| Threads/block | 256 |
| Blocks | 4096 |
| Repeat reads/thread | 64 |
| Warm-up launches | 3 |
| Iterations | 100 |

Run 2:

```text
./cache_behavior 2097152 256 64 100
```

| Setting | Value |
| ------- | ----: |
| Output elements | 2097152 |
| Largest input | 256.000 MiB |
| Output bytes | 8.000 MiB |
| Threads/block | 256 |
| Blocks | 8192 |
| Repeat reads/thread | 64 |
| Warm-up launches | 3 |
| Iterations | 100 |

## Access Pattern

Each output element is a sum of repeated reads from a bounded working set:

```text
base        = output_index & (working_set_elements - 1)
input_index = (base + repeat_index * 131) & (working_set_elements - 1)
output[i]   = sum(input[input_index])
```

The working-set sizes are powers of two so the mask operation wraps the
index back into the active working set.

## Measurement Table

| Output elements | Working set | Working-set elements | Repeat reads/thread | Avg time (ms) | Useful bandwidth (GB/s) | Result |
| --------------: | ----------- | -------------------: | ------------------: | ------------: | ----------------------: | ------ |
|         1048576 | 4 KiB       |                 1024 |                  64 |         0.814 |                 335.101 | PASS   |
|         1048576 | 64 KiB      |                16384 |                  64 |         0.288 |                 947.586 | PASS   |
|         1048576 | 1 MiB       |               262144 |                  64 |         0.321 |                 848.445 | PASS   |
|         1048576 | 8 MiB       |              2097152 |                  64 |         0.335 |                 812.993 | PASS   |
|         1048576 | 64 MiB      |             16777216 |                  64 |         0.333 |                 818.997 | PASS   |
|         1048576 | 256 MiB     |             67108864 |                  64 |         0.328 |                 832.011 | PASS   |
|         2097152 | 4 KiB       |                 1024 |                  64 |         1.700 |                 320.832 | PASS   |
|         2097152 | 64 KiB      |                16384 |                  64 |         0.620 |                 879.555 | PASS   |
|         2097152 | 1 MiB       |               262144 |                  64 |         0.657 |                 830.366 | PASS   |
|         2097152 | 8 MiB       |              2097152 |                  64 |         0.663 |                 822.773 | PASS   |
|         2097152 | 64 MiB      |             16777216 |                  64 |         0.664 |                 821.184 | PASS   |
|         2097152 | 256 MiB     |             67108864 |                  64 |         0.665 |                 820.383 | PASS   |

Recorded from:

```text
std_record.log
```

## Detailed Timing

| Output elements | Working set | Min time (ms) | Max time (ms) | Avg time (ms) |
| --------------: | ----------- | ------------: | ------------: | ------------: |
|         1048576 | 4 KiB       |         0.731 |         0.901 |         0.814 |
|         1048576 | 64 KiB      |         0.281 |         0.318 |         0.288 |
|         1048576 | 1 MiB       |         0.307 |         0.335 |         0.321 |
|         1048576 | 8 MiB       |         0.334 |         0.340 |         0.335 |
|         1048576 | 64 MiB      |         0.326 |         0.338 |         0.333 |
|         1048576 | 256 MiB     |         0.326 |         0.341 |         0.328 |
|         2097152 | 4 KiB       |         1.619 |         1.784 |         1.700 |
|         2097152 | 64 KiB      |         0.610 |         0.640 |         0.620 |
|         2097152 | 1 MiB       |         0.641 |         0.666 |         0.657 |
|         2097152 | 8 MiB       |         0.659 |         0.670 |         0.663 |
|         2097152 | 64 MiB      |         0.660 |         0.677 |         0.664 |
|         2097152 | 256 MiB     |         0.661 |         0.673 |         0.665 |

## Observation Questions

1.
2.
3.

## Observation

The `64 KiB` case was the fastest in both runs:

- `1048576` outputs: 947.586 GB/s
- `2097152` outputs: 879.555 GB/s

The smallest `4 KiB` working set was unexpectedly slower:

- `1048576` outputs: 335.101 GB/s
- `2097152` outputs: 320.832 GB/s

For the larger working-set labels, bandwidth stayed fairly flat:

- `1048576` outputs: about 813 to 848 GB/s for `1 MiB` through `256 MiB`
- `2097152` outputs: about 820 to 830 GB/s for `1 MiB` through `256 MiB`

The second run roughly doubles the output size, and most kernel times also
roughly double. That means the benchmark is scaling predictably with the
number of output elements.

## Interpretation

The raw bandwidth numbers are "useful bandwidth", not DRAM bandwidth. The
calculation counts every repeated read as useful traffic, even when some
of those reads are served from cache. That is why values can exceed the
GTX 1080 Ti's theoretical DRAM bandwidth of about 484 GB/s.

The original hypothesis was that very small working sets should be fastest
and very large working sets should eventually slow down. This run does not
show that simple curve.

The `4 KiB` case may be slower because many threads repeatedly touch a
very small set of addresses, creating pressure on the same cache sets or
memory pipelines. The result is still cache-friendly, but not necessarily
the highest-throughput access pattern.

The larger working-set labels are also a reminder to check the address
formula carefully. When `working_set_elements` is larger than
`output_elements`, this kernel does not necessarily touch the entire
larger allocation. For example, with `1048576` output elements, the base
index covers about the first `1 MiB` floats, plus the repeat offset range.
So the `64 MiB` and `256 MiB` labels allocate a larger possible working
set, but the measured active region is smaller than the label suggests.

The next version of this experiment should force the kernel to sample
across the whole working-set size before using it as a strict cache
capacity benchmark.

## Connection To Previous Labs

The memory-coalescing lab controlled the shape of the warp's global-memory
addresses. This cache-behavior lab keeps the access formula regular and
changes the amount of data being reused.

Together, the two labs separate two different memory questions:

- coalescing: are neighboring threads asking for compact addresses?
- caching: are repeated reads hitting data that is still nearby in cache?

## Next Experiment

The next Week 2 step is Experiment 03 - Shared Memory Bank Conflicts:

- shared-memory access patterns
- bank conflicts
- global memory versus shared memory reuse

Start here:

```text
../03_shared_memory_bank_conflicts
```
