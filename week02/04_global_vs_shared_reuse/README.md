# Experiment 04 - Global Reuse vs Shared Reuse

## Objective

Compare repeated direct global-memory reads against explicitly staging
reused data in shared memory.

The previous labs separated three memory ideas:

- memory coalescing: neighboring lanes should access compact global
  addresses
- cache behavior: repeated reads may be served by hardware-managed cache
- bank conflicts: shared memory is fast, but its bank structure matters

This lab combines them in a small stencil-style workload where neighboring
threads reuse overlapping input values.

## First Question

When a block repeatedly uses the same neighborhood of input values, is it
faster to read from global memory directly or stage the values in shared
memory first?

## Hypothesis

Direct global reads may already perform well when accesses are coalesced
and cache reuse is strong.

Shared-memory staging should help when the same input values are reused
many times by neighboring threads, but it also adds overhead:

- extra instructions to load the tile
- one or more `__syncthreads()`
- shared-memory capacity pressure
- possible bank conflicts if the shared-memory layout is poor

So the expected result is not simply "shared memory always wins." The
measurement should show when explicit staging is worth the overhead.

## Recommended Reading

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  read the shared-memory matrix multiplication examples and the sections
  on memory optimization workflow.
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/):
  practical refresher for declaring shared memory, loading tiles, and
  synchronizing a block.
- [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/):
  useful for understanding why shared-memory layout and padding matter.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135):
  modern research example of reducing memory traffic by explicitly
  organizing reuse.
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785):
  use the arithmetic-intensity idea to explain when reducing memory
  traffic should matter.

## Implementation TODOs

This lab is intentionally left as a starter exercise.

Fill in the TODO sections in:

```text
global_vs_shared_reuse.cu
```

Required implementation work:

1. Implement `stencil_global_direct`.
2. Implement `stencil_shared_tiled`.
3. Use the same mathematical result in both kernels.
4. Keep the benchmark harness unchanged so the comparison is fair.

The benchmark harness, timing logic, result verification, and reporting
are already provided.

The program may compile before the TODOs are complete, but the benchmark
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
nvcc -O3 -std=c++17 -arch=sm_61 global_vs_shared_reuse.cu -o global_vs_shared_reuse
```

## Run

Default:

```bash
./global_vs_shared_reuse
```

Arguments:

```bash
./global_vs_shared_reuse <elements> <threads_per_block> <radius> <repeat_accesses> <iterations>
```

Examples:

```bash
./global_vs_shared_reuse 16777216 256 4 32 100
./global_vs_shared_reuse 33554432 256 4 32 100
./global_vs_shared_reuse 16777216 256 8 64 100
```

Make shortcut:

```bash
make run
```

## Workload

Each output element sums a small neighborhood repeatedly:

```text
for repeat in repeat_accesses:
    for offset in 0..radius:
        sum += input[i + offset]
output[i] = sum
```

Neighboring threads reuse overlapping input elements. For example, with
`radius = 4`:

```text
thread i     reads input[i + 0] ... input[i + 4]
thread i + 1 reads input[i + 1] ... input[i + 5]
```

The shared-memory version should stage a block tile plus halo:

```text
shared elements = threads_per_block + radius + padding
```

The padding value is an unused tail slot in this 1D lab. It is included as
practice for later 2D shared-memory layouts where padding can change bank
mapping.

## Suggested Measurements

| Kernel | Padding | Radius | Repeat accesses | Shared memory/block | Avg time (ms) | Logical bandwidth (GB/s) | Result |
| ------ | ------: | -----: | --------------: | ------------------: | ------------: | -----------------------: | ------ |
| direct_global | 0 | 4 | 32 | 0 KiB | | | |
| shared_tiled | 0 | 4 | 32 | | | | |
| shared_padded | 1 | 4 | 32 | | | | |

## Detailed Timing

| Kernel | Padding | Min time (ms) | Max time (ms) | Avg time (ms) |
| ------ | ------: | ------------: | ------------: | ------------: |
| direct_global | 0 | | | |
| shared_tiled | 0 | | | |
| shared_padded | 1 | | | |

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


## Connection To Previous Labs

Vector addition showed a mostly streaming global-memory workload.

Matrix multiplication showed that shared-memory tiling can improve reuse.

The bank-conflict lab showed that shared memory can still be slow if its
address pattern is poor.

This lab asks when explicit shared-memory staging is actually better than
letting coalesced global loads and hardware cache do the work.

## Next Experiment

After this lab, Week 2 can branch in two directions:

- use Nsight Compute to collect memory counters for the Week 2 kernels
- move into Week 3 performance modeling with arithmetic intensity and
  Roofline analysis
