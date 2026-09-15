# Experiment 05 - Nsight Compute Memory Counters

## Objective

Use Nsight Compute to explain Week 2 memory experiments with hardware
counters instead of timing alone.

The previous labs measured elapsed time and derived bandwidth. This lab
asks the next question: when a kernel gets slower or faster, what changed
inside the GPU memory system?

## First Question

Can Nsight Compute counters explain the differences between direct global
reads, shared-memory staging, and shared-memory bank-conflict patterns?

## Hypothesis

Timing tells us what happened, but profiler counters should help explain
why it happened.

For memory-focused kernels, useful counters should show differences in:

- achieved occupancy
- DRAM throughput
- L1/TEX and L2 throughput
- global-load efficiency
- shared-memory throughput
- warp stall reasons

## Recommended Reading

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html):
  the main reference for understanding sections, metrics, reports, and
  replay behavior.
- [Nsight Compute CLI Documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html):
  use this when running `ncu` from the terminal and exporting reports.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  read the profiling and memory optimization workflow sections before
  interpreting counters.
- [Hierarchical Roofline Analysis: How to Collect Data using Performance Tools on Intel CPUs and NVIDIA GPUs](https://arxiv.org/abs/2009.02449):
  practical paper for connecting profiler counters to cache-level
  performance explanations.
- [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis](https://arxiv.org/abs/2501.12084):
  stretch reading; focus on how the authors choose counters and isolate
  one hardware feature at a time.

## Implementation TODOs

This lab is a profiling exercise, so there is no new CUDA kernel to fill
in.

Required work:

1. Build the previous Week 2 executables.
2. Run Nsight Compute on the target kernels.
3. Record timing and selected memory counters.
4. Explain whether the counters support your timing-based interpretation.

## Build

Build the target labs first:

```bash
cd ../03_shared_memory_bank_conflicts
make

cd ../04_global_vs_shared_reuse
make
```

Then return to this directory:

```bash
cd ../05_nsight_compute_memory_counters
```

## Run

Profile the global-vs-shared reuse lab:

```bash
make profile-global-shared
```

Profile the shared-memory bank-conflict lab:

```bash
make profile-bank-conflicts
```

Equivalent direct command:

```bash
ncu --set detailed --target-processes all ./target_program args...
```

## Suggested Measurements

Fill this table from the Nsight Compute report.

| Kernel | Case | Avg time (ms) | DRAM throughput | L2 throughput | Shared throughput | Occupancy | Main stall reason |
| ------ | ---- | ------------: | --------------: | ------------: | ----------------: | --------: | ----------------- |
| global_vs_shared_reuse | direct_global | | | | | | |
| global_vs_shared_reuse | shared_tiled | | | | | | |
| global_vs_shared_reuse | shared_padded | | | | | | |
| shared_bank_conflicts | stride 1 | | | | | | |
| shared_bank_conflicts | stride 32 | | | | | | |

## Counter Notes

The exact metric names can vary across GPU architecture and Nsight Compute
version. Prefer the report's section names first:

- Speed Of Light
- Memory Workload Analysis
- Scheduler Statistics
- Warp State Statistics
- Launch Statistics
- Occupancy

Then write down the most useful counters you see in those sections.

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


## Connection To Previous Labs

The Week 2 timing labs answered which versions were faster.

This lab starts answering why:

- Was the kernel limited by DRAM?
- Was it limited by shared-memory behavior?
- Did occupancy change?
- Did the warp stall reason change?

## Next Experiment

After collecting memory counters, move into Week 3 performance modeling:

```text
../../week03/01_arithmetic_intensity
```
