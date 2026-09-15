# Experiment 03 - Shared Memory Bank Conflicts

## Objective

Measure how shared-memory address patterns affect throughput when a warp
reads from shared memory.

The cache-behavior lab looked at hardware-managed cache reuse. This lab
switches to programmer-managed shared memory and asks a more local
question: what happens when neighboring lanes in a warp hit the same
shared-memory bank?

## First Question

How much performance is lost when shared-memory reads change from
conflict-free access to 2-way, 4-way, 8-way, 16-way, and 32-way bank
conflicts?

## Hypothesis

Stride 1 should be fastest because neighboring lanes access neighboring
shared-memory banks.

Power-of-two strides should become slower as the conflict degree grows,
because multiple lanes in the same warp need service from the same bank.

## Recommended Reading

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html):
  read the shared-memory and memory-hierarchy sections as the reference
  model for shared-memory behavior.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  focus on shared memory, bank conflicts, and the matrix multiplication
  optimization examples.
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/):
  practical introduction to declaring, indexing, and synchronizing shared
  memory.
- [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/):
  shows why padding shared-memory arrays can remove bank conflicts.
- [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis](https://arxiv.org/abs/2501.12084):
  stretch reading; skim the microbenchmarking methodology to see how
  architecture papers isolate one hardware behavior at a time.

## Implementation TODOs

This lab is intentionally left as a starter exercise.

Fill in the TODO section in:

```text
shared_bank_conflicts.cu
```

Required implementation work:

1. Compute the global output index and whether this thread is in range.
2. Compute the shared-memory index for this thread:

   ```text
   shared_index = threadIdx.x * stride
   ```

3. Store one input value into `shared_values[shared_index]`.
4. Synchronize the block.
5. Repeatedly read from `shared_values[shared_index]`, accumulate the
   value, and write the sum to output.

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
nvcc -O3 -std=c++17 -arch=sm_61 shared_bank_conflicts.cu -o shared_bank_conflicts
```

## Run

Default:

```bash
./shared_bank_conflicts
```

Arguments:

```bash
./shared_bank_conflicts <elements> <threads_per_block> <repeat_accesses> <iterations>
```

Examples:

```bash
./shared_bank_conflicts 16777216 256 256 100
./shared_bank_conflicts 33554432 256 256 100
```

Make shortcut:

```bash
make run
```

## Access Pattern

Each thread stores one value into shared memory and then repeatedly reads
that same shared-memory location:

```text
shared_index = threadIdx.x * stride
shared_values[shared_index] = input[global_index]
sum += shared_values[shared_index]
output[global_index] = sum
```

For 32-bit floats, the shared-memory bank is roughly:

```text
bank = shared_index % 32
```

That means:

- stride 1: neighboring lanes use neighboring banks
- stride 2: two lanes map to each bank
- stride 4: four lanes map to each bank
- stride 32: all lanes in a warp map to the same bank

## Suggested Measurements

| Stride | Expected conflict | Shared memory/block | Avg time (ms) | Useful bandwidth (GB/s) | Result |
| -----: | ----------------: | ------------------: | ------------: | ----------------------: | ------ |
|      1 |             1-way |                     |               |                         |        |
|      2 |             2-way |                     |               |                         |        |
|      4 |             4-way |                     |               |                         |        |
|      8 |             8-way |                     |               |                         |        |
|     16 |            16-way |                     |               |                         |        |
|     32 |            32-way |                     |               |                         |        |

## Detailed Timing

| Stride | Min time (ms) | Max time (ms) | Avg time (ms) |
| -----: | ------------: | ------------: | ------------: |
|      1 |               |               |               |
|      2 |               |               |               |
|      4 |               |               |               |
|      8 |               |               |               |
|     16 |               |               |               |
|     32 |               |               |               |

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


## Connection To Previous Labs

Memory coalescing studied the addresses requested by neighboring threads
in global memory.

Cache behavior studied whether repeated reads can be served from nearby
hardware-managed memory.

This lab studies a different part of the hierarchy: shared memory is fast
and explicitly managed, but it still has structure. A bad shared-memory
index pattern can serialize a warp even when the data is already on-chip.

## Next Experiment

After bank conflicts, compare global-memory reuse against explicit
shared-memory reuse:

- direct repeated global reads
- shared-memory staging
- padding to avoid shared-memory bank conflicts
- when shared memory helps and when it just adds overhead
