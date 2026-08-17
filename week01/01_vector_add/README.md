# Experiment 01 — Vector Addition

## Objective

Implement a simple CUDA vector-addition kernel and use it to establish
a baseline for GPU programming and performance measurement.

The experiment is intentionally simple so that the relationship between:

- threads
- blocks
- global memory
- kernel execution
- memory bandwidth

can be studied clearly.

---

## Hardware

GPU:

NVIDIA GeForce GTX 1080 Ti

Architecture:

Pascal

Compute Capability:

6.1 (`sm_61`)

VRAM:

11 GiB

Theoretical memory bandwidth:

~484 GB/s

---

## Software

OS:

Ubuntu 24.04 LTS

Kernel:

6.17.0-19-generic

NVIDIA Driver:

580.173.02

CUDA Toolkit:

12.6.3

NVCC:

12.6.85

GCC:

13.3.0

---

## Kernel

The kernel performs:

    c[i] = a[i] + b[i]

Each CUDA thread processes one element.

Thread index:

    i = blockIdx.x * blockDim.x + threadIdx.x

---

## Configuration

Vector size:

    N = 16,777,216

Elements:

    16,777,216 floats

Bytes per vector:

    64 MiB

Threads per block:

    256

Blocks:

    65,536

Total threads:

    16,777,216

Compilation target:

    sm_61

Compilation command:

    nvcc -arch=sm_61 vector_add.cu -o vector_add

---

## Correctness

Result:

    PASS

All output elements were verified to be:

    c[i] = 3.0

---

## Memory Traffic

For every element:

    read a[i]  = 4 bytes
    read b[i]  = 4 bytes
    write c[i] = 4 bytes

Total:

    12 bytes / element

For 16,777,216 elements:

    16,777,216 × 12 bytes
    = 201,326,592 bytes
    = 192 MiB

Therefore the kernel moves approximately:

    192 MiB

of data.

---

## Initial Measurement

First observed kernel time:

    0.978496 ms

This measurement was significantly slower than subsequent executions.

This suggests that the first measurement should not be treated as a
steady-state performance measurement.

---

## Repeated Measurements

10 executions:

    0.543584 ms
    0.588768 ms
    0.592000 ms
    0.593920 ms
    0.598144 ms
    0.544864 ms
    0.548384 ms
    0.555232 ms
    0.589344 ms
    0.592992 ms

Approximate mean:

    0.5757 ms

Approximate effective bandwidth:

    ~334 GB/s

Fastest observed bandwidth:

    ~353 GB/s

Theoretical GTX 1080 Ti bandwidth:

    ~484 GB/s

---

## Initial Observation

The vector-addition kernel performs very little computation:

    c[i] = a[i] + b[i]

For every 12 bytes of memory traffic, only one floating-point addition
is performed.

Therefore the kernel is expected to be primarily memory-bandwidth
bound rather than compute bound.

Measured bandwidth is significantly below the theoretical DRAM
bandwidth of the GTX 1080 Ti.

---

## Next Experiment

Build a proper GPU microbenchmark.

The benchmark should:

1. Initialize CUDA once.
2. Warm up the kernel.
3. Execute the kernel many times.
4. Measure steady-state execution time.
5. Calculate min / max / average time.
6. Calculate effective memory bandwidth.
7. Record the results.

After establishing a reliable baseline, investigate:

- threads per block
- memory access patterns
- vector size
- cache effects
- memory coalescing

## Proper Microbenchmark Baseline

Warm-up:
    1 kernel launch

Measured iterations:
    100

Threads per block:
    256

Blocks:
    65,536

Results:

    Min kernel time:
        0.521216 ms

    Max kernel time:
        0.541248 ms

    Average kernel time:
        0.524786 ms

Effective bandwidth:

    383.635 GB/s

Theoretical bandwidth:

    ~484 GB/s

Approximate bandwidth efficiency:

    ~79.3%

Observation:

The repeated benchmark is substantially more stable than the
initial single-run measurement. The kernel achieves approximately
79% of the theoretical DRAM bandwidth of the GTX 1080 Ti.

## Experiment 1C — Threads Per Block

### Hypothesis

Increasing threads per block may improve performance by providing
more warps that can be scheduled while other warps are waiting on
memory.

### Fixed Variables

- GPU: GTX 1080 Ti
- N: 16,777,216
- Kernel: vector addition
- Iterations: 100
- Memory access pattern: sequential
- Compiler target: sm_61

### Results

| Threads/block | Blocks | Avg time (ms) | Effective BW (GB/s) |
| ------------- | ------ | ------------- | ------------------- |
| 32            | 524288 | 0.815390      | 246.908             |
| 64            | 262144 | 0.518549      | 388.250             |
| 128           | 131072 | 0.519136      | 387.811             |
| 256           | 65536  | 0.519689      | 387.398             |
| 512           | 32768  | 0.518992      | 387.919             |
| 1024          | 16384  | 0.517872      | 388.758             |

### Observation

32 threads/block performs substantially worse than all other tested
configurations.

Performance improves dramatically when moving from one warp per block
to two warps per block.

Above 64 threads/block, performance is effectively flat at around
388 GB/s.

### Current Hypothesis

The 32-thread configuration may have insufficient independent warp
work to effectively hide memory latency.

However, this hypothesis has not yet been proven.

Further experiments and hardware profiling are required.

### Experiment 1D: Stride-2 Access

Configuration:

- GPU: GeForce GTX 1080 Ti
- N: 16,777,216
- Threads/block: 256
- Iterations: 100
- Compiler: CUDA 12.6, `-arch=sm_61`

Baseline contiguous access:

- Average kernel time: ~0.520 ms
- Effective bandwidth: ~388 GB/s

Stride-2 access:

- Average kernel time: ~0.782 ms
- Effective bandwidth: ~258 GB/s

Observation:

The stride-2 version was substantially slower than the contiguous
version, despite processing fewer useful elements in the current
implementation.

Interpretation:

This suggests that the memory access pattern has a significant effect
on performance. However, this experiment is not yet an apples-to-apples
comparison because the stride-2 version writes only half as many
elements. A fair experiment will keep the amount of useful work
constant while changing only the input access pattern.

## Memory Access Stride Experiment

### Setup

- GPU: NVIDIA GeForce GTX 1080 Ti
- Architecture: Pascal / GP102
- Compute Capability: 6.1
- N: 16,777,216
- Threads/block: 256
- Iterations: 100
- Operation: C[i] = A[i * stride] + B[i * stride]

### Results

| Stride | Avg Kernel Time | Effective Bandwidth |
| -----: | --------------: | ------------------: |
|      1 |       ~0.520 ms |           ~388 GB/s |
|      2 |       ~0.881 ms |           ~229 GB/s |
|      4 |       ~1.602 ms |           ~126 GB/s |
|      8 |       ~3.036 ms |          ~66.3 GB/s |

### Observation

Increasing the memory access stride significantly reduces effective
bandwidth. The kernel performs the same logical amount of useful work,
but threads within a warp access increasingly separated addresses.

The experiment demonstrates the performance impact of non-contiguous
global-memory access on GPU kernels.

### Caveat

This benchmark measures effective bandwidth, not the exact number of
DRAM transactions. Hardware-level transaction behavior was not directly
profiled because the available Nsight Compute version does not support
GP102 performance counters.


---

## Questions

1. Why is vector addition memory-bandwidth bound?

2. Why is measured bandwidth lower than theoretical peak bandwidth?

3. How much variation exists between repeated measurements?

4. Does increasing the number of threads per block change performance?

5. Does changing the memory access pattern change performance?

6. How does memory coalescing affect bandwidth?

7. How does cache behavior affect the result?

8. What percentage of theoretical bandwidth can this simple kernel
   realistically achieve?