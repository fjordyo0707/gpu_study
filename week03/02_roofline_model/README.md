# Experiment 02 - Roofline Model

## Objective

Build a simple Roofline model from your own benchmark results.

Arithmetic intensity tells you how many FLOPs a kernel performs per byte
of memory traffic. Roofline combines that with hardware ceilings to ask:

```text
Is this kernel limited by memory bandwidth or compute throughput?
```

## First Question

Can a simple Roofline model explain the performance differences between
vector add, matrix multiplication, and the arithmetic-intensity sweep?

## Hypothesis

Low-arithmetic-intensity kernels should sit near the memory-bandwidth
roof.

High-arithmetic-intensity kernels should move toward the compute roof, but
only if they have enough parallelism, instruction mix, occupancy, and data
reuse to use the hardware well.

## Recommended Reading

- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785):
  the core paper for this lab.
- [Hierarchical Roofline Analysis: How to Collect Data using Performance Tools on Intel CPUs and NVIDIA GPUs](https://arxiv.org/abs/2009.02449):
  practical guide for extending the simple model to cache-level ceilings.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html):
  use this when comparing hand-calculated Roofline points with profiler
  counters.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  review effective bandwidth, throughput, and optimization methodology.
- [8 Steps to 3.7 TFLOP/s on NVIDIA V100 GPU: Roofline Analysis and Other Tricks](https://arxiv.org/abs/2008.11326):
  readable optimization case study using Roofline analysis.

## Implementation TODOs

This is an analysis lab instead of a CUDA-kernel lab.

Fill in:

```text
roofline_template.csv
```

Required work:

1. Add benchmark rows from previous labs.
2. Fill measured arithmetic intensity.
3. Fill measured GFLOP/s.
4. Run the analysis script.
5. Decide whether each kernel is memory-bound, compute-bound, or below
   both roofs for another reason.

## Hardware Constants

For the GTX 1080 Ti, use conservative starting ceilings:

```text
Theoretical memory bandwidth: 484 GB/s
Approx FP32 peak:             10600 GFLOP/s
```

These are starting points, not truth carved in silicon. Later you can
replace them with measured STREAM-like bandwidth and a measured compute
microbenchmark peak.

## Run

```bash
make analyze
```

Equivalent manual command:

```bash
python3 roofline_from_csv.py roofline_template.csv
```

The script writes:

```text
roofline_report.md
```

## Input CSV

Columns:

```text
kernel,arithmetic_intensity_flop_per_byte,measured_gflops,notes
```

Example rows:

```text
vector_add,0.083,350,low arithmetic intensity
matmul_tiled,?,1006,fill arithmetic intensity estimate
```

## Suggested Rows

| Kernel | Source lab | Arithmetic intensity | GFLOP/s | Classification |
| ------ | ---------- | -------------------: | ------: | -------------- |
| vector_add | week01/01_vector_add | | | |
| matmul_naive | week01/02_matrix_multiply | | | |
| matmul_tiled | week01/02_matrix_multiply | | | |
| matmul_register_blocking | week01/02_matrix_multiply | | | |
| arithmetic_intensity fma=1 | week03/01_arithmetic_intensity | | | |
| arithmetic_intensity fma=64 | week03/01_arithmetic_intensity | | | |
| arithmetic_intensity fma=1024 | week03/01_arithmetic_intensity | | | |

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


## Connection To Previous Labs

The earlier labs measured isolated effects:

- vector add: low arithmetic intensity
- matrix multiplication: higher arithmetic intensity and reuse
- memory hierarchy labs: memory behavior details
- arithmetic intensity sweep: controlled FLOP/byte changes

Roofline is the first model that puts those points on one map.

## Next Experiment

After building a simple Roofline model, continue with bottleneck analysis:

```text
../03_bottleneck_analysis
```
