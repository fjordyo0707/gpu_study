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

## Shared Memory Bank Model

For this lab, use this mental model:

```text
shared-memory banks per SM = 32
warp lanes                 = 32
```

The 32 banks are part of the SM's shared-memory system. They are not
private to one warp. A warp also has 32 lanes, and when that warp executes
a shared-memory instruction, those lanes issue their shared-memory address
requests together.

The ideal case is that the 32 lanes hit 32 different banks:

```text
warp lane:  0   1   2   3   ... 31
bank:       0   1   2   3   ... 31
```

Then the bank requests can be served in parallel.

If multiple lanes request different addresses in the same bank, that bank
must serve those requests in multiple steps. That is a bank conflict.

For NVIDIA CUDA GPUs, 32 shared-memory banks is the standard model for
modern architectures, including this lab's GTX 1080 Ti / Pascal GPU. The
exact behavior can vary across architectures because bank width, broadcast
rules, multicast behavior, and conflict handling have changed over time,
but `32 banks` is the right starting point for these experiments.

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

## Implementation Status

This lab has been completed and recorded in:

```text
std_record.log
```

The critical implementation work was in:

```text
shared_bank_conflicts.cu
```

Implemented kernel tasks:

1. Compute the global output index and whether this thread is in range.
2. Compute the shared-memory index for this thread:

   ```text
   shared_index = threadIdx.x * stride
   ```

3. Store one input value into `shared_values[shared_index]`.
4. Synchronize the block.
5. Repeatedly read from `shared_values[shared_index]`, accumulate the
   value, and write the sum to output.

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

## Recorded Runs

Run 1:

```text
./shared_bank_conflicts 16777216 256 256 100
```

| Setting | Value |
| ------- | ----: |
| Elements | 16777216 |
| Input bytes | 64.000 MiB |
| Output bytes | 64.000 MiB |
| Threads/block | 256 |
| Blocks | 65536 |
| Repeat accesses | 256 |
| Warm-up launches | 3 |
| Iterations | 100 |

Run 2:

```text
./shared_bank_conflicts 33554432 256 256 100
```

| Setting | Value |
| ------- | ----: |
| Elements | 33554432 |
| Input bytes | 128.000 MiB |
| Output bytes | 128.000 MiB |
| Threads/block | 256 |
| Blocks | 131072 |
| Repeat accesses | 256 |
| Warm-up launches | 3 |
| Iterations | 100 |

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

## Measurement Table

| Elements | Stride | Expected conflict | Shared memory/block | Avg time (ms) | Useful bandwidth (GB/s) | Result |
| -------: | -----: | ----------------: | ------------------: | ------------: | ----------------------: | ------ |
| 16777216 |      1 |             1-way |           1.000 KiB |         1.272 |               13606.793 | PASS   |
| 16777216 |      2 |             2-way |           1.996 KiB |         1.268 |               13654.336 | PASS   |
| 16777216 |      4 |             4-way |           3.988 KiB |         1.277 |               13563.397 | PASS   |
| 16777216 |      8 |             8-way |           7.973 KiB |         1.289 |               13430.336 | PASS   |
| 16777216 |     16 |            16-way |          15.941 KiB |         1.293 |               13388.629 | PASS   |
| 16777216 |     32 |            32-way |          31.879 KiB |         1.736 |                9974.350 | PASS   |
| 33554432 |      1 |             1-way |           1.000 KiB |         2.642 |               13106.189 | PASS   |
| 33554432 |      2 |             2-way |           1.996 KiB |         2.619 |               13223.507 | PASS   |
| 33554432 |      4 |             4-way |           3.988 KiB |         2.629 |               13172.527 | PASS   |
| 33554432 |      8 |             8-way |           7.973 KiB |         2.613 |               13251.316 | PASS   |
| 33554432 |     16 |            16-way |          15.941 KiB |         2.701 |               12820.762 | PASS   |
| 33554432 |     32 |            32-way |          31.879 KiB |         3.396 |               10196.330 | PASS   |

Recorded from:

```text
std_record.log
```

## Detailed Timing

| Elements | Stride | Min time (ms) | Max time (ms) | Avg time (ms) |
| -------: | -----: | ------------: | ------------: | ------------: |
| 16777216 |      1 |         1.261 |         1.286 |         1.272 |
| 16777216 |      2 |         1.236 |         1.294 |         1.268 |
| 16777216 |      4 |         1.259 |         1.296 |         1.277 |
| 16777216 |      8 |         1.260 |         1.315 |         1.289 |
| 16777216 |     16 |         1.271 |         1.321 |         1.293 |
| 16777216 |     32 |         1.630 |         1.795 |         1.736 |
| 33554432 |      1 |         2.119 |         3.258 |         2.642 |
| 33554432 |      2 |         2.515 |         2.673 |         2.619 |
| 33554432 |      4 |         2.469 |         2.693 |         2.629 |
| 33554432 |      8 |         2.517 |         2.693 |         2.613 |
| 33554432 |     16 |         2.540 |         3.888 |         2.701 |
| 33554432 |     32 |         3.234 |         3.738 |         3.396 |

## Observation Questions

1.
2.
3.

## Observation

The middle strides did not form a clean monotonic slowdown:

- For `16777216` elements, strides 1 through 16 stayed close together:
  1.268 to 1.293 ms.
- For `33554432` elements, strides 1 through 16 also stayed close:
  2.613 to 2.701 ms.

Stride 32 was the clear outlier:

- `16777216` elements: 1.272 ms at stride 1 vs 1.736 ms at stride 32,
  about 1.36x slower.
- `33554432` elements: 2.642 ms at stride 1 vs 3.396 ms at stride 32,
  about 1.29x slower.

Useful bandwidth follows the same pattern. The first run drops from
13606.793 GB/s at stride 1 to 9974.350 GB/s at stride 32. The second run
drops from 13106.189 GB/s at stride 1 to 10196.330 GB/s at stride 32.

## Interpretation

The high-level bank-conflict hypothesis was partly supported: the strongest
expected conflict, stride 32, was clearly slower.

However, the result does not show a smooth 1-way, 2-way, 4-way, 8-way,
16-way, 32-way staircase. That means this benchmark is measuring more than
only bank conflicts.

Two details matter:

1. Shared-memory usage grows with stride. Stride 1 uses about 1 KiB per
   block, while stride 32 uses about 31.879 KiB per block. The stride 32
   case can reduce occupancy because each block reserves much more shared
   memory.
2. The kernel repeatedly reads the same shared-memory address for each
   thread. The compiler or hardware may keep some of that value close to
   the thread after the first read, so repeated accesses may not expose
   bank conflicts as strongly as a dependent shared-memory access pattern.

So the safe conclusion is:

- stride 32 is bad in this benchmark
- stride 1 through 16 are very similar here
- this benchmark is good for seeing a severe shared-memory pattern, but
  it is not a perfectly isolated bank-conflict microbenchmark

A stricter follow-up would keep shared-memory allocation constant across
cases and use a dependent or volatile shared-memory load pattern so each
repeat is forced to issue as a shared-memory access.

## Connection To Previous Labs

Memory coalescing studied the addresses requested by neighboring threads
in global memory.

Cache behavior studied whether repeated reads can be served from nearby
hardware-managed memory.

This lab studies a different part of the hierarchy: shared memory is fast
and explicitly managed, but it still has structure. A bad shared-memory
index pattern can serialize a warp even when the data is already on-chip.

## Next Experiment

The next Week 2 step is Experiment 04 - Global Reuse vs Shared Reuse:

- direct repeated global reads
- shared-memory staging
- padding to avoid shared-memory bank conflicts
- when shared memory helps and when it just adds overhead

Start here:

```text
../04_global_vs_shared_reuse
```
