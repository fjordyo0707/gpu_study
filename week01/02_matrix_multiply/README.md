# Experiment 02 - Matrix Multiplication

## Objective

Implement a naive CUDA matrix-multiplication kernel and use it to
compare a higher-arithmetic-intensity workload against vector addition.

Vector addition performed one floating-point operation for every
12 bytes of useful memory traffic, so it was strongly memory-bandwidth
bound.

Matrix multiplication performs many multiply-add operations for each
input element. Even a naive implementation gives us a better experiment
for asking:

- When does a kernel become compute-bound?
- How much work does each thread perform?
- How does global-memory reuse affect performance?
- What does GFLOP/s tell us that bandwidth does not?

## Kernel

The kernel computes square matrix multiplication:

```text
C[row][col] = sum(A[row][k] * B[k][col])
```

Each CUDA thread computes one output element.

Thread mapping:

```text
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

## First Question

Is naive matrix multiplication limited mainly by memory bandwidth, or by
floating-point throughput?

## Hypothesis

Compared with vector addition, naive matrix multiplication should reach
much higher arithmetic intensity because each output element performs
`N` multiply-add steps.

However, the naive kernel reads from global memory inside the inner loop
without using shared memory tiling, so it may still waste a large amount
of memory bandwidth.

## Build

```bash
make
```

This builds all matrix-multiplication lab executables:

- `matmul_naive`
- `matmul_tiled`
- `matmul_tile_sweep`
- `matmul_register_blocking`
- `matmul_memory_layout`
- `matmul_cublas`

The naive baseline can also be built manually:

```bash
nvcc -O3 -std=c++17 -arch=sm_61 matmul_naive.cu -o matmul_naive
```

## Run

Default:

```bash
./matmul_naive
```

Arguments:

```bash
./matmul_naive <matrix_size> <block_dim> <iterations>
```

Examples:

```bash
./matmul_naive 512 16 20
./matmul_naive 1024 16 20
./matmul_naive 2048 16 10
```

The second argument is both the block width and block height. For
example, `16` means a `16 x 16` thread block.

## Metrics

For an `N x N` matrix multiplication:

```text
Output elements = N * N
Work per output = N multiplies + N adds
Approx FLOPs    = 2 * N * N * N
```

The benchmark reports:

- minimum kernel time
- maximum kernel time
- average kernel time
- achieved GFLOP/s
- correctness

## Suggested Measurements

Keep the GPU, compiler, and matrix values fixed. Change one variable at
a time.

| N    | Block dim | Iterations | Avg time (ms) | GFLOP/s | Result |
| ---: | --------: | ---------: | ------------: | ------: | ------ |
|  512 |        16 |         20 |         0.614 | 436.875 | PASS   |
| 1024 |        16 |         20 |         4.792 | 448.103 | PASS   |
| 2048 |        16 |         10 |        39.023 | 440.248 | not recorded |

Recorded from:

```text
std_record.log
```

Detailed timing:

| N    | Matrix bytes | Grid dim  | Min time (ms) | Max time (ms) | Avg time (ms) |
| ---: | -----------: | --------- | ------------: | ------------: | ------------: |
|  512 |      1.0 MiB | 32 x 32   |         0.609 |         0.621 |         0.614 |
| 1024 |      4.0 MiB | 64 x 64   |         4.733 |         4.825 |         4.792 |
| 2048 |     16.0 MiB | 128 x 128 |        38.626 |        39.327 |        39.023 |

The later tile-size sweep records the measured `8 x 8`, `16 x 16`, and
`32 x 32` shared-memory block cases.

## Observation

The naive matrix-multiplication kernel explicitly records PASS for the
`512 x 512` and `1024 x 1024` runs. The `2048 x 2048` timing is recorded,
but the raw log does not include its final correctness line.

Throughput is stable across the size sweep:

- `512 x 512`: 436.875 GFLOP/s
- `1024 x 1024`: 448.103 GFLOP/s
- `2048 x 2048`: 440.248 GFLOP/s

Doubling `N` increases the amount of work by approximately 8x, and the
measured runtime also increases by approximately 8x:

- `512 -> 1024`: 0.614 ms to 4.792 ms
- `1024 -> 2048`: 4.792 ms to 39.023 ms

This is consistent with the expected `O(N^3)` work of matrix
multiplication.

## Interpretation

The naive kernel achieves much higher arithmetic throughput than the
vector-add experiment, but it is still far below the GTX 1080 Ti's peak
single-precision throughput.

The key issue is data reuse. The mathematical operation has high reuse,
but the naive kernel does not explicitly capture that reuse. Each thread
loads values from global memory inside the inner loop, and neighboring
threads repeatedly load overlapping matrix data.

This makes the result a good baseline. The next question is whether
shared-memory tiling can reduce redundant global-memory traffic and move
the kernel closer to the GPU's compute capability.

## Analysis Prompts

1. How does achieved GFLOP/s compare with vector-add bandwidth results?
2. Does increasing `N` improve or reduce performance?
3. Does changing block size matter?
4. What memory access pattern does each warp use for matrix `A`?
5. What memory access pattern does each warp use for matrix `B`?
6. Which values are reused by different threads?
7. Why might shared memory help this kernel?

## Optimization Roadmap

| Step | Kernel idea | Main change | Status |
| ---- | ----------- | ----------- | ------ |
| 02A  | Naive matrix multiplication | One thread computes one output element using global memory | Complete |
| 02B  | Shared-memory tiled matrix multiplication | Cache tiles of `A` and `B` in shared memory | Complete |
| 02C  | Tile-size sweep | Compare different tile sizes and block shapes | Complete |
| 02D  | Register blocking | Compute multiple output values per thread | Complete |
| 02E  | Memory-layout experiment | Compare normal `B` access with transposed or reordered `B` | Complete |
| 02F  | cuBLAS comparison | Compare custom kernels against `cublasSgemm` | Complete |

## Experiment Coverage

The log contains six matrix-multiplication experiment groups:

| Log section | Source file | What it measures | Written below |
| ----------- | ----------- | ---------------- | ------------- |
| `matmul_naive` | `matmul_naive.cu` | naive global-memory baseline | yes |
| `matmul_tiled` | `matmul_tiled.cu` | shared-memory tiled baseline | yes |
| `matmul_tile_sweep` | `matmul_tile_sweep.cu` | `8 x 8`, `16 x 16`, and `32 x 32` tile sizes | yes |
| `matmul_register_blocking` | `matmul_register_blocking.cu` | two output values per thread | yes |
| `matmul_memory_layout` | `matmul_memory_layout.cu` | normal `B` vs pre-transposed `B` | yes |
| `matmul_cublas` | `matmul_cublas.cu` | custom tiled kernel vs `cublasSgemm` | yes |

## Comparison Summary

All rows below use the `1024 x 1024` measurements from `std_record.log`.

| Experiment | Kernel or variant | Avg time (ms) | GFLOP/s | Comparison |
| ---------- | ----------------- | ------------: | ------: | ---------- |
| Naive baseline | `matmul_naive` | 4.792 | 448.103 | baseline |
| Shared memory | `matmul_tiled` | 2.133 | 1006.580 | 2.25x faster than naive |
| Tile sweep | `8 x 8` tile | 2.976 | 721.548 | slower than `16 x 16` and `32 x 32` |
| Tile sweep | `16 x 16` tile | 2.369 | 906.614 | middle result in sweep |
| Tile sweep | `32 x 32` tile | 2.160 | 994.135 | fastest tile-sweep result |
| Register blocking | 2 outputs/thread | 1.564 | 1373.441 | 1.36x faster than tiled baseline |
| Memory layout | normal `B` | 4.927 | 435.837 | baseline for layout test |
| Memory layout | transposed `B` | 18.258 | 117.619 | 3.71x slower than normal `B` |
| cuBLAS comparison | custom tiled | 2.125 | 1010.346 | custom kernel in same program |
| cuBLAS comparison | `cublasSgemm` | 0.284 | 7559.304 | 7.48x faster than custom tiled |

The strongest custom teaching kernel is register blocking at
1373.441 GFLOP/s. The fastest overall result is `cublasSgemm` at
7559.304 GFLOP/s.

## Executable Usage

All commands in this section should be run from:

```bash
week01/02_matrix_multiply
```

Build the full lab:

```bash
make
```

### Naive Baseline

File:

```text
matmul_naive.cu
```

Usage:

```bash
./matmul_naive <matrix_size> <block_dim> <iterations>
```

Example:

```bash
./matmul_naive 1024 16 20
```

Purpose:

Establish the baseline one-output-element-per-thread matrix
multiplication result using global memory directly.

Make shortcut:

```bash
make run-naive
```

### Shared-Memory Tiled Baseline

File:

```text
matmul_tiled.cu
```

Usage:

```bash
./matmul_tiled <matrix_size> <tile_dim> <iterations>
```

Example:

```bash
./matmul_tiled 1024 16 20
```

Purpose:

Measure the effect of caching tiles of `A` and `B` in shared memory
before accumulating partial dot products.

Make shortcut:

```bash
make run-tiled
```

### Tile-Size Sweep

File:

```text
matmul_tile_sweep.cu
```

Usage:

```bash
./matmul_tile_sweep <matrix_size> <iterations> [tile_dim...]
```

Example:

```bash
./matmul_tile_sweep 1024 20 8 16 32
```

Purpose:

Run the shared-memory tiled kernel with multiple tile sizes so the
runtime, throughput, shared-memory usage, and block shape can be
compared directly.

Make shortcut:

```bash
make run-tile-sweep
```

### Register Blocking

File:

```text
matmul_register_blocking.cu
```

Usage:

```bash
./matmul_register_blocking <matrix_size> <tile_dim> <iterations>
```

Example:

```bash
./matmul_register_blocking 1024 16 20
```

Purpose:

Compute two output columns per thread so one loaded `A` value can be
reused across multiple accumulators in registers.

Make shortcut:

```bash
make run-register-blocking
```

### Memory Layout

File:

```text
matmul_memory_layout.cu
```

Usage:

```bash
./matmul_memory_layout <matrix_size> <block_dim> <iterations>
```

Example:

```bash
./matmul_memory_layout 1024 16 20
```

Purpose:

Compare a normal `B[k][col]` access pattern against a pre-transposed
`B` layout to study how contiguous access changes performance.

Make shortcut:

```bash
make run-memory-layout
```

### cuBLAS Comparison

File:

```text
matmul_cublas.cu
```

Usage:

```bash
./matmul_cublas <matrix_size> <tile_dim> <iterations>
```

Example:

```bash
./matmul_cublas 1024 16 20
```

Purpose:

Compare the custom shared-memory tiled kernel against NVIDIA's
`cublasSgemm` implementation.

Make shortcut:

```bash
make run-cublas
```

### Recording Logs

Append experiment output to the lab log:

```bash
./matmul_tile_sweep 1024 20 8 16 32 | tee -a std_record.log
./matmul_register_blocking 1024 16 20 | tee -a std_record.log
./matmul_memory_layout 1024 16 20 | tee -a std_record.log
./matmul_cublas 1024 16 20 | tee -a std_record.log
```

## Optimization 02B - Shared Memory Tiling

### Goal

Implement a tiled matrix-multiplication kernel using shared memory.

The kernel should load a tile of `A` and a tile of `B` into shared
memory, synchronize the block, compute partial sums, and repeat until
the output element is complete.

### Kernel Plan

```text
for each tile:
    load A tile into shared memory
    load B tile into shared memory
    synchronize
    accumulate partial dot product
    synchronize
write C[row][col]
```

### Fixed Variables

- GPU: NVIDIA GeForce GTX 1080 Ti
- Architecture: Pascal / GP102
- CUDA Toolkit: 12.6.3
- Compiler target: `sm_61`
- Matrix values: `A = 1.0`, `B = 2.0`
- Timing method: CUDA events
- Warm-up launches: 3

### Build

```bash
nvcc -O3 -std=c++17 -arch=sm_61 matmul_tiled.cu -o matmul_tiled
```

### Run

```bash
./matmul_tiled 512 16 20
./matmul_tiled 1024 16 20
./matmul_tiled 2048 16 20
```

### Measurement Table

| N    | Tile dim | Iterations | Avg time (ms) | GFLOP/s | Speedup vs naive | Result |
| ---: | -------: | ---------: | ------------: | ------: | ---------------: | ------ |
|  512 |       16 |         20 |         0.280 | 960.037 |            2.19x | PASS   |
| 1024 |       16 |         20 |         2.133 | 1006.580 |           2.25x | PASS   |
| 2048 |       16 |         20 |        17.968 | 956.163 |            2.17x | PASS   |

Recorded from:

```text
std_record.log
```

### Detailed Timing

| N    | Tile dim | Grid dim | Min time (ms) | Max time (ms) | Avg time (ms) |
| ---: | -------: | -------- | ------------: | ------------: | ------------: |
|  512 |       16 | 32 x 32  |         0.278 |         0.285 |         0.280 |
| 1024 |       16 | 64 x 64  |         2.129 |         2.141 |         2.133 |
| 2048 |       16 | 128 x 128 |       17.040 |        19.557 |        17.968 |

Shared memory per block:

```text
2.000 KiB
```

### Observation Questions

1.
2.
3.

### Observation

The shared-memory tiled kernel passes correctness for all tested sizes.

Compared with the naive kernel, the tiled kernel is consistently faster:

- `512 x 512`: 0.614 ms to 0.280 ms, or 2.19x faster
- `1024 x 1024`: 4.792 ms to 2.133 ms, or 2.25x faster
- `2048 x 2048`: 39.023 ms to 17.968 ms, or 2.17x faster

The tiled kernel reaches roughly 0.96-1.01 TFLOP/s across the tested
sizes. Throughput is still fairly stable as `N` increases, but the
`2048 x 2048` run shows a wider min-to-max spread than the smaller
cases.

### Interpretation

Shared-memory tiling significantly improves performance because each
block reuses tiles of `A` and `B` instead of repeatedly loading all
values directly from global memory inside every thread's inner loop.

The result confirms that the naive kernel was wasting bandwidth through
redundant global-memory traffic. The tiled kernel is still far below
peak GPU throughput, which motivates the tile-size sweep and later
per-thread work experiments below.

## Optimization 02C - Tile-Size Sweep

### Goal

Measure how tile size changes runtime, throughput, occupancy, and shared
memory usage.

### Measurement Table

| N    | Tile dim | Threads/block | Shared memory/block | Avg time (ms) | GFLOP/s | Result |
| ---: | -------: | ------------: | ------------------: | ------------: | ------: | ------ |
| 1024 |        8 |            64 |           0.500 KiB |         2.976 | 721.548 | PASS   |
| 1024 |       16 |           256 |           2.000 KiB |         2.369 | 906.614 | PASS   |
| 1024 |       32 |          1024 |           8.000 KiB |         2.160 | 994.135 | PASS   |

Recorded from:

```text
std_record.log
```

### Detailed Timing

| Tile dim | Grid dim  | Min time (ms) | Max time (ms) | Avg time (ms) |
| -------: | --------- | ------------: | ------------: | ------------: |
|        8 | 128 x 128 |         2.943 |         3.193 |         2.976 |
|       16 | 64 x 64   |         2.222 |         2.522 |         2.369 |
|       32 | 32 x 32   |         2.127 |         2.212 |         2.160 |

### Observation Questions

1.
2.
3.

### Observation

For `1024 x 1024`, increasing tile size improved performance across the
tested range:

- `8 x 8`: 721.548 GFLOP/s
- `16 x 16`: 906.614 GFLOP/s
- `32 x 32`: 994.135 GFLOP/s

The `32 x 32` tile was fastest in this sweep, reaching nearly the same
throughput as the earlier `matmul_tiled` baseline.

### Interpretation

Larger tiles increase data reuse because each loaded tile element can be
used by more multiply-add operations inside the block.

The tradeoff is that larger tiles also use more threads per block and
more shared memory per block. The `32 x 32` case uses 1024 threads per
block, which is the CUDA maximum, so future optimizations may need to
improve per-thread work rather than simply increasing block size.

## Optimization 02D - Register Blocking

### Goal

Have each thread compute more than one output element so that values
loaded from memory can be reused in registers.

### Measurement Table

| N    | Tile dim | Outputs/thread | Avg time (ms) | GFLOP/s | Speedup vs tiled | Result |
| ---: | -------: | -------------: | ------------: | ------: | ---------------: | ------ |
| 1024 |       16 |              2 |         1.564 | 1373.441 |          1.36x | PASS   |

Recorded from:

```text
std_record.log
```

### Detailed Timing

| N    | Tile dim | Outputs/thread | Grid dim | Shared memory/block | Min time (ms) | Max time (ms) | Avg time (ms) |
| ---: | -------: | -------------: | -------- | ------------------: | ------------: | ------------: | ------------: |
| 1024 |       16 |              2 | 32 x 64  |           3.000 KiB |         1.559 |         1.574 |         1.564 |

### Observation Questions

1.
2.
3.

### Observation

Register blocking improved the `1024 x 1024` result from the tiled
baseline's 2.133 ms to 1.564 ms.

Throughput increased from about 1006.580 GFLOP/s for the tiled baseline
to 1373.441 GFLOP/s.

### Interpretation

Computing two output values per thread lets each thread reuse one loaded
`A` value across two accumulators. This increases the useful work done
per thread and reduces some redundant memory traffic.

The improvement suggests that the previous tiled kernel was not only
limited by global-memory traffic. Per-thread instruction mix, register
reuse, and block scheduling also matter.

## Optimization 02E - Memory Layout

### Goal

Change how matrix `B` is stored or accessed and measure whether improved
memory locality changes performance.

### Measurement Table

| N    | Kernel variant | B layout   | Avg time (ms) | GFLOP/s | Speedup vs normal | Result |
| ---: | -------------- | ---------- | ------------: | ------: | ----------------: | ------ |
| 1024 | normal_b       | row-major  |         4.927 | 435.837 |             1.00x | PASS   |
| 1024 | transposed_b   | transposed |        18.258 | 117.619 |             0.27x | PASS   |

Recorded from:

```text
std_record.log
```

### Detailed Timing

| Kernel variant | Min time (ms) | Max time (ms) | Avg time (ms) |
| -------------- | ------------: | ------------: | ------------: |
| normal_b       |         4.887 |         4.963 |         4.927 |
| transposed_b   |        17.187 |        20.246 |        18.258 |

### Observation Questions

1.
2.
3.

### Observation

The pre-transposed `B` layout was much slower for this thread mapping.

The normal layout reached 435.837 GFLOP/s, while the transposed layout
fell to 117.619 GFLOP/s.

### Interpretation

This result is a useful correction to the first intuition that
"contiguous" data is always better.

In the normal `B[k][col]` layout, neighboring threads with adjacent
`col` values read adjacent `B` addresses for a fixed `k`. In the
transposed layout, those same neighboring threads read addresses spaced
by `N`, which produces a much worse warp-level access pattern.

The layout must be evaluated together with the thread mapping.

## Optimization 02F - cuBLAS Comparison

### Goal

Compare the custom kernels against NVIDIA's optimized SGEMM
implementation.

### Measurement Table

| N    | Kernel | Avg time (ms) | GFLOP/s | Speedup vs naive | Speedup vs tiled | Result |
| ---: | ------ | ------------: | ------: | ---------------: | ---------------: | ------ |
| 1024 | naive baseline |         4.792 | 448.103 |            1.00x |            0.45x | PASS   |
| 1024 | custom_tiled   |         2.125 | 1010.346 |           2.25x |            1.00x | PASS   |
| 1024 | cublasSgemm    |         0.284 | 7559.304 |          16.87x |            7.51x | PASS   |

Recorded from:

```text
std_record.log
```

### Detailed Timing

| Kernel | Min time (ms) | Max time (ms) | Avg time (ms) |
| ------ | ------------: | ------------: | ------------: |
| custom_tiled |         2.118 |         2.141 |         2.125 |
| cublasSgemm  |         0.274 |         0.299 |         0.284 |

### Observation Questions

1.
2.
3.

### Observation

cuBLAS is dramatically faster than the custom kernels.

For `1024 x 1024`, `cublasSgemm` achieved 7559.304 GFLOP/s, compared
with 1010.346 GFLOP/s for the custom tiled kernel measured in the same
program.

### Interpretation

The custom kernels demonstrate the main memory-optimization ideas, but
they are still simple teaching kernels.

cuBLAS is much closer to a production SGEMM implementation. It likely
uses deeper blocking, more careful register tiling, instruction
scheduling, vectorized memory movement, and architecture-specific tuning
that this lab has not implemented yet.

## Overall Observations

The matrix-multiplication experiments show a clear performance ladder:

| Kernel | Avg time for 1024 x 1024 (ms) | Throughput (GFLOP/s) |
| ------ | ----------------------------: | -------------------: |
| Naive |                         4.792 |              448.103 |
| Shared-memory tiled |          2.133 |             1006.580 |
| Register blocking |             1.564 |             1373.441 |
| cuBLAS |                        0.284 |             7559.304 |

Shared-memory tiling proves that the naive kernel wastes global-memory
traffic. Caching tiles of `A` and `B` lets each block reuse data and more
than doubles throughput.

Register blocking shows that optimization does not stop at shared
memory. By computing multiple output values per thread, the kernel gets
more useful arithmetic from values already held in registers. This moves
the discussion from only global-memory bandwidth toward instruction
scheduling, register reuse, occupancy, and per-thread arithmetic work.

The memory-layout experiment is the most useful warning. Pre-transposing
`B` made the kernel much slower, even though the data looked more
contiguous from a single thread's point of view. What matters is the
warp-level access pattern. With the normal layout, neighboring threads
with adjacent `col` values read adjacent `B` addresses for a fixed `k`.
With the transposed layout, neighboring threads read addresses separated
by `N`, which hurts coalescing.

cuBLAS gives the target direction. The best teaching kernel in this lab
reaches about 1.37 TFLOP/s, while `cublasSgemm` reaches about
7.56 TFLOP/s for the same matrix size. That gap suggests cuBLAS is using
deeper blocking, register tiling, architecture-specific scheduling, and
better instruction and memory pipelining.

The main lesson is that matrix multiplication optimization is a stack of
reuse decisions:

1. Reuse data across threads with shared memory.
2. Reuse data inside a thread with registers.
3. Preserve coalesced warp-level memory access.
4. Tune the work shape for the actual GPU architecture.

This makes Week 2's memory-hierarchy work the natural next step,
especially warp-level coalescing and cache behavior.
