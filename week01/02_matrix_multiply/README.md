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

Equivalent manual command:

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
| 2048 |        16 |         10 |        39.023 | 440.248 | PASS   |

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

Then test block shape:

| N    | Block dim | Iterations | Avg time (ms) | GFLOP/s | Result |
| ---: | --------: | ---------: | ------------: | ------: | ------ |
| 1024 |         8 |         20 |               |         |        |
| 1024 |        16 |         20 |               |         |        |
| 1024 |        32 |         20 |               |         |        |

## Observation

The naive matrix-multiplication kernel passes correctness for all tested
sizes.

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
| 02B  | Shared-memory tiled matrix multiplication | Cache tiles of `A` and `B` in shared memory | Next |
| 02C  | Tile-size sweep | Compare different tile sizes and block shapes | Future |
| 02D  | Register blocking | Compute multiple output values per thread | Future |
| 02E  | Memory-layout experiment | Compare normal `B` access with transposed or reordered `B` | Future |
| 02F  | cuBLAS comparison | Compare custom kernels against `cublasSgemm` | Future |

## Optimization 02B Template - Shared Memory Tiling

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

- GPU:
- Architecture:
- CUDA Toolkit:
- Compiler target:
- Matrix values:
- Timing method:
- Warm-up launches:

### Build

```bash
nvcc -O3 -std=c++17 -arch=sm_61 matmul_tiled.cu -o matmul_tiled
```

### Run

```bash
./matmul_tiled 512 16 20
./matmul_tiled 1024 16 20
./matmul_tiled 2048 16 10
```

### Measurement Table

| N    | Tile dim | Iterations | Avg time (ms) | GFLOP/s | Speedup vs naive | Result |
| ---: | -------: | ---------: | ------------: | ------: | ---------------: | ------ |
|  512 |          |            |               |         |                  |        |
| 1024 |          |            |               |         |                  |        |
| 2048 |          |            |               |         |                  |        |

### Detailed Timing

| N    | Tile dim | Grid dim | Min time (ms) | Max time (ms) | Avg time (ms) |
| ---: | -------: | -------- | ------------: | ------------: | ------------: |
|  512 |          |          |               |               |               |
| 1024 |          |          |               |               |               |
| 2048 |          |          |               |               |               |

### Observation Questions

1.
2.
3.

### Observation


### Interpretation


## Future Optimization Template - Tile-Size Sweep

### Goal

Measure how tile size changes runtime, throughput, occupancy, and shared
memory usage.

### Measurement Table

| N    | Tile dim | Threads/block | Shared memory/block | Avg time (ms) | GFLOP/s | Result |
| ---: | -------: | ------------: | ------------------: | ------------: | ------: | ------ |
| 1024 |          |               |                     |               |         |        |
| 1024 |          |               |                     |               |         |        |
| 1024 |          |               |                     |               |         |        |

### Observation Questions

1.
2.
3.

### Observation


### Interpretation


## Future Optimization Template - Register Blocking

### Goal

Have each thread compute more than one output element so that values
loaded from memory can be reused in registers.

### Measurement Table

| N    | Tile dim | Outputs/thread | Avg time (ms) | GFLOP/s | Speedup vs tiled | Result |
| ---: | -------: | -------------: | ------------: | ------: | ---------------: | ------ |
| 1024 |          |                |               |         |                  |        |
| 1024 |          |                |               |         |                  |        |
| 2048 |          |                |               |         |                  |        |

### Observation Questions

1.
2.
3.

### Observation


### Interpretation


## Future Optimization Template - Memory Layout

### Goal

Change how matrix `B` is stored or accessed and measure whether improved
memory locality changes performance.

### Measurement Table

| N    | Kernel variant | B layout | Avg time (ms) | GFLOP/s | Speedup vs tiled | Result |
| ---: | -------------- | -------- | ------------: | ------: | ---------------: | ------ |
| 1024 |                |          |               |         |                  |        |
| 1024 |                |          |               |         |                  |        |
| 2048 |                |          |               |         |                  |        |

### Observation Questions

1.
2.
3.

### Observation


### Interpretation


## Future Optimization Template - cuBLAS Comparison

### Goal

Compare the custom kernels against NVIDIA's optimized SGEMM
implementation.

### Measurement Table

| N    | Kernel | Avg time (ms) | GFLOP/s | Speedup vs naive | Speedup vs tiled | Result |
| ---: | ------ | ------------: | ------: | ---------------: | ---------------: | ------ |
| 1024 |        |               |         |                  |                  |        |
| 1024 |        |               |         |                  |                  |        |
| 2048 |        |               |         |                  |                  |        |

### Observation Questions

1.
2.
3.

### Observation


### Interpretation
