# Experiment 04 - Bottleneck Analysis

## Objective

Practice turning benchmark results into a bottleneck diagnosis.

Roofline gives a first classification, but real kernels can sit below both
the memory roof and the compute roof. This lab teaches the next step:
forming a testable hypothesis about why a kernel is slow.

## First Question

When a kernel is slower than the simple Roofline model predicts, what is the
most likely bottleneck and what experiment should we run next?

## Hypothesis

A useful bottleneck diagnosis should connect three things:

1. The measured symptom.
2. The likely hardware or software reason.
3. A small next experiment that can confirm or reject the explanation.

The goal is not to guess perfectly. The goal is to make the next experiment
specific enough that the result teaches you something.

## Recommended Reading

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  use the optimization workflow sections as the main model for this lab.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html):
  useful when the bottleneck might be launch overhead, synchronization, or
  host/device transfer behavior.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html):
  read the Speed Of Light, Memory Workload Analysis, and Warp State ideas
  even if modern `ncu` cannot profile the Pascal GPU directly.
- [Hierarchical Roofline Analysis: How to Collect Data using Performance Tools on Intel CPUs and NVIDIA GPUs](https://arxiv.org/abs/2009.02449):
  practical guide for connecting cache-level measurements to bottleneck
  explanations.

## Implementation TODOs

This is an analysis lab.

Fill in:

```text
bottleneck_template.csv
```

Required work:

1. Pick several kernels from Week 1 through Week 3.
2. Copy their measured arithmetic intensity, GFLOP/s, and bandwidth.
3. Write one observed symptom for each kernel.
4. Write one first hypothesis for the bottleneck.
5. Choose one next experiment that could test the hypothesis.
6. Run the report script.

## Run

```bash
make analyze
```

Equivalent manual command:

```bash
python3 bottleneck_from_csv.py bottleneck_template.csv
```

The script writes:

```text
bottleneck_report.md
```

## Bottleneck Hints

Use these categories as a starting point:

| Symptom | Possible bottleneck | Next experiment |
| ------- | ------------------- | --------------- |
| Low AI and bandwidth near measured peak | DRAM bandwidth | Improve coalescing or reuse |
| Low AI and bandwidth far below measured peak | Bad access pattern or overhead | Check stride/coalescing/timeline |
| High AI and low GFLOP/s | Instruction dependency or low occupancy | Change dependency pattern or block size |
| High variance between min/max time | System noise or unstable clocks | Increase iterations and repeat runs |
| Kernel time tiny but total program slow | Launch or copy overhead | Use Nsight Systems timeline |
| Shared-memory case slower than global-memory case | Synchronization or bank behavior | Vary tile size, padding, and reuse count |

## Suggested Rows

| Kernel | Source lab | Symptom | First hypothesis | Next experiment |
| ------ | ---------- | ------- | ---------------- | --------------- |
| vector_add | week01/01_vector_add | | | |
| matmul_naive | week01/02_matrix_multiply | | | |
| matmul_tiled | week01/02_matrix_multiply | | | |
| arithmetic_intensity_fma_1 | week03/01_arithmetic_intensity | | | |
| arithmetic_intensity_fma_1024 | week03/01_arithmetic_intensity | | | |
| memory_stream | week03/03_memory_vs_compute | | | |
| compute_chain_512 | week03/03_memory_vs_compute | | | |

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


## Connection To Previous Labs

The earlier labs measured isolated effects. This lab asks you to combine
them:

- memory coalescing explains some bandwidth symptoms
- shared memory explains some reuse and synchronization symptoms
- arithmetic intensity explains memory-bound vs compute-bound behavior
- Roofline explains the first-order performance limit

## Next Experiment

After building a bottleneck diagnosis, compare it with whatever profiling
signals your local setup can provide:

```text
../05_performance_counter_workflow
```
