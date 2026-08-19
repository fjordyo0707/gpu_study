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

## Implementation TODOs

This lab is intentionally left as a starter exercise.

Fill in the TODO sections in:

```text
memory_coalescing.cu
```

Required implementation work:

1. Compute the global thread index in `copy_offset`.
2. Implement the offset access pattern.
3. Compute the global thread index in `copy_stride`.
4. Implement the strided access pattern.

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
| offset  |         0 |           256 |               |                         |        |
| offset  |         1 |           256 |               |                         |        |
| offset  |         2 |           256 |               |                         |        |
| offset  |         4 |           256 |               |                         |        |
| offset  |         8 |           256 |               |                         |        |
| stride  |         2 |           256 |               |                         |        |
| stride  |         4 |           256 |               |                         |        |
| stride  |         8 |           256 |               |                         |        |

## Detailed Timing

| Pattern | Parameter | Min time (ms) | Max time (ms) | Avg time (ms) |
| ------- | --------: | ------------: | ------------: | ------------: |
| offset  |         0 |               |               |               |
| offset  |         1 |               |               |               |
| offset  |         2 |               |               |               |
| offset  |         4 |               |               |               |
| offset  |         8 |               |               |               |
| stride  |         2 |               |               |               |
| stride  |         4 |               |               |               |
| stride  |         8 |               |               |               |

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


## Connection To Previous Labs

The vector-add stride experiment showed bandwidth falling as stride
increased.

The matrix-multiply memory-layout experiment showed that pre-transposing
`B` made performance worse for the current thread mapping.

This lab should explain both results using the same idea: performance
depends on the memory addresses accessed by neighboring threads in the
same warp.

## Next Experiment

After memory coalescing, measure cache behavior:

- repeated reads from the same working set
- working-set size sweep
- L2 cache effects
- cache reuse vs streaming access
