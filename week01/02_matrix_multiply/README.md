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
|  512 |        16 |         20 |               |         |        |
| 1024 |        16 |         20 |               |         |        |
| 2048 |        16 |         10 |               |         |        |

Then test block shape:

| N    | Block dim | Iterations | Avg time (ms) | GFLOP/s | Result |
| ---: | --------: | ---------: | ------------: | ------: | ------ |
| 1024 |         8 |         20 |               |         |        |
| 1024 |        16 |         20 |               |         |        |
| 1024 |        32 |         20 |               |         |        |

## Analysis Prompts

1. How does achieved GFLOP/s compare with vector-add bandwidth results?
2. Does increasing `N` improve or reduce performance?
3. Does changing block size matter?
4. What memory access pattern does each warp use for matrix `A`?
5. What memory access pattern does each warp use for matrix `B`?
6. Which values are reused by different threads?
7. Why might shared memory help this kernel?

## Next Experiment

After recording the naive baseline, implement a tiled matrix
multiplication kernel using shared memory.

The goal will be to reduce redundant global-memory loads and measure how
much data reuse improves performance.
