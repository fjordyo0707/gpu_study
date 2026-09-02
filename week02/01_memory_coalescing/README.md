# Experiment 01 - Memory Coalescing

## Objective

Measure how warp-level global-memory access patterns affect effective
memory bandwidth.

Week 1 already showed two important clues:

- strided vector access reduced bandwidth
- transposing `B` in matrix multiplication made performance worse

This lab focuses directly on the reason: neighboring threads in a warp
must access neighboring memory addresses for efficient coalescing.

## First Question

How much performance is lost when a warp changes from contiguous access
to offset or strided access?

## Hypothesis

Contiguous access should achieve the highest useful bandwidth because
threads in a warp access adjacent memory locations.

Small offsets may be slightly slower if the warp's memory request crosses
additional memory segments.

Strided access should become progressively slower because neighboring
threads touch addresses farther apart, increasing wasted memory traffic
per useful float copied.

## Recommended Reading

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  read the coalesced global-memory access sections, especially simple,
  misaligned, and strided access patterns.
- [How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/):
  a direct companion for this lab's offset and stride measurements.
- [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/):
  connects coalescing to matrix layout and shows how shared memory can
  reorder strided global accesses.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html):
  use the memory workload analysis sections later to confirm whether
  uncoalesced accesses increase memory transactions.
- [Hierarchical Roofline Analysis: How to Collect Data using Performance Tools on Intel CPUs and NVIDIA GPUs](https://arxiv.org/abs/2009.02449):
  a practical paper for learning how memory hierarchy effects show up in
  measured counters, not just wall-clock time.
- [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis](https://arxiv.org/abs/2501.12084):
  recent architecture paper; skim the memory subsystem and global-memory
  access parts to see how researchers design microbenchmarks like this
  lab.

## Implementation Status

This lab has been completed and recorded in:

```text
std_record.log
```

The critical implementation work was in:

```text
memory_coalescing.cu
```

Implemented kernel tasks:

1. Compute the global thread index in `copy_offset`.
2. Implement the offset access pattern.
3. Compute the global thread index in `copy_stride`.
4. Implement the strided access pattern.

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
nvcc -O3 -std=c++17 -arch=sm_61 memory_coalescing.cu -o memory_coalescing
```

## Run

Default:

```bash
./memory_coalescing
```

Arguments:

```bash
./memory_coalescing <elements> <threads_per_block> <iterations>
```

Examples:

```bash
./memory_coalescing 16777216 256 100
./memory_coalescing 33554432 256 100
```

Make shortcut:

```bash
make run
```

Default measured cases:

- `offset 0`
- `offset 1`
- `offset 2`
- `offset 4`
- `offset 8`
- `stride 2`
- `stride 4`
- `stride 8`

## Recorded Run

```text
./memory_coalescing 16777216 256 100
```

| Setting | Value |
| ------- | ----: |
| Elements | 16777216 |
| Input bytes | 512.000 MiB |
| Output bytes | 64.000 MiB |
| Threads/block | 256 |
| Blocks | 65536 |
| Warm-up launches | 3 |
| Iterations | 100 |

## Metrics

The benchmark copies one useful float per thread:

```text
output[i] = input[index]
```

Useful traffic per element:

```text
read input[index] = 4 bytes
write output[i]   = 4 bytes
total             = 8 bytes / element
```

The reported bandwidth is useful bandwidth:

```text
effective GB/s = useful bytes / average kernel time
```

For strided access, useful bandwidth is not the same as actual DRAM
transaction bandwidth. The gap is the point of the experiment.

## Access Patterns

### Contiguous

```text
output[i] = input[i]
```

Neighboring threads read neighboring floats.

### Offset

```text
output[i] = input[i + offset]
```

Neighboring threads still read neighboring floats, but the warp starts at
a shifted address.

### Strided

```text
output[i] = input[i * stride]
```

Neighboring threads read addresses separated by `stride` floats.

## Suggested Measurements

| Pattern | Parameter | Threads/block | Avg time (ms) | Useful bandwidth (GB/s) | Result |
| ------- | --------: | ------------: | ------------: | ----------------------: | ------ |
| offset  |         0 |           256 |         0.373 |                 359.749 | PASS   |
| offset  |         1 |           256 |         0.380 |                 352.894 | PASS   |
| offset  |         2 |           256 |         0.380 |                 353.056 | PASS   |
| offset  |         4 |           256 |         0.380 |                 353.304 | PASS   |
| offset  |         8 |           256 |         0.395 |                 339.938 | PASS   |
| stride  |         2 |           256 |         0.572 |                 234.573 | PASS   |
| stride  |         4 |           256 |         0.934 |                 143.735 | PASS   |
| stride  |         8 |           256 |         1.713 |                  78.364 | PASS   |

Recorded from:

```text
std_record.log
```

## Detailed Timing

| Pattern | Parameter | Min time (ms) | Max time (ms) | Avg time (ms) |
| ------- | --------: | ------------: | ------------: | ------------: |
| offset  |         0 |         0.371 |         0.374 |         0.373 |
| offset  |         1 |         0.379 |         0.389 |         0.380 |
| offset  |         2 |         0.379 |         0.382 |         0.380 |
| offset  |         4 |         0.378 |         0.389 |         0.380 |
| offset  |         8 |         0.394 |         0.399 |         0.395 |
| stride  |         2 |         0.570 |         0.584 |         0.572 |
| stride  |         4 |         0.932 |         0.936 |         0.934 |
| stride  |         8 |         1.711 |         1.716 |         1.713 |

## Observation Questions

1.
2.
3.

## Observation

Offset accesses stayed close to the contiguous baseline:

- `offset 0`: 359.749 GB/s
- `offset 1`: 352.894 GB/s
- `offset 2`: 353.056 GB/s
- `offset 4`: 353.304 GB/s
- `offset 8`: 339.938 GB/s

The small offset cases are only slightly slower because neighboring
threads still read neighboring floats. The warp may start at a less
convenient address, but the access pattern is still mostly coalesced.

Strided accesses became much slower as the stride increased:

- `stride 2`: 234.573 GB/s
- `stride 4`: 143.735 GB/s
- `stride 8`: 78.364 GB/s

Compared with `offset 0`, `stride 8` keeps only about 21.8% of the useful
bandwidth.

## Interpretation

This result shows that coalescing depends on the addresses requested by
neighboring threads in a warp, not just on how much useful data the kernel
copies.

For offset access, lane 0 reads `input[offset]`, lane 1 reads
`input[offset + 1]`, lane 2 reads `input[offset + 2]`, and so on. The warp
still asks for a compact range of memory.

For strided access, lane 0 reads `input[0]`, lane 1 reads `input[stride]`,
lane 2 reads `input[2 * stride]`, and so on. The useful values are spread
across a wider memory region, so the GPU needs more memory transactions
to collect the same number of useful floats.

That is why useful bandwidth drops from 359.749 GB/s for contiguous reads
to 78.364 GB/s for `stride 8`.

## Connection To Previous Labs

The vector-add stride experiment showed bandwidth falling as stride
increased.

The matrix-multiply memory-layout experiment showed that pre-transposing
`B` made performance worse for the current thread mapping.

This lab should explain both results using the same idea: performance
depends on the memory addresses accessed by neighboring threads in the
same warp.

## Next Experiment

The next Week 2 step is Experiment 02 - Cache Behavior:

- repeated reads from the same working set
- working-set size sweep
- L2 cache effects
- cache reuse vs streaming access

Start here:

```text
../02_cache_behavior
```
