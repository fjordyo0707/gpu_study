# Experiment 05 - Performance Counter Workflow

## Objective

Connect the Week 3 performance model to profiler signals.

On a newer GPU, this lab would lean heavily on Nsight Compute counters and
Roofline sections. On this repo's GTX 1080 Ti / Pascal GPU, modern Nsight
Compute may not support the device, so this lab uses a layered workflow:

- required: CUDA-event timing and derived metrics
- recommended: Nsight Systems timeline summaries
- optional: Nsight Compute / `nvprof` / CUPTI only if your local setup
  supports them

## First Question

Which profiler or measurement signals would confirm a memory-bound,
compute-bound, or overhead-bound diagnosis?

## Hypothesis

Even without detailed hardware counters, a useful profiling workflow can
still test the bottleneck analysis from the previous lab.

The expected signal should match the diagnosis:

- memory-bound kernels: high effective bandwidth and low arithmetic
  intensity
- compute-bound kernels: high GFLOP/s and lower effective bandwidth
  because time is spent on arithmetic
- overhead-bound kernels: timeline gaps, short kernels, copies, or
  synchronization dominate the program

## Recommended Reading

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html):
  practical reference for timeline capture and `nsys stats`.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html):
  read the Speed Of Light, Roofline, Scheduler, and Memory Workload
  sections as the ideal counter-based workflow.
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/):
  use `cuobjdump` or `nvdisasm` later when you want to inspect generated
  instructions.
- [CUPTI Documentation](https://docs.nvidia.com/cupti/index.html):
  stretch reading for understanding how profiling tools collect activity
  and metric data.
- [Hierarchical Roofline Analysis: How to Collect Data using Performance Tools on Intel CPUs and NVIDIA GPUs](https://arxiv.org/abs/2009.02449):
  explains how profiler data can support Roofline analysis across memory
  hierarchy levels.

## Implementation TODOs

This is a workflow lab instead of a new CUDA-kernel lab.

Fill in:

```text
counter_notes_template.md
```

Required work:

1. Pick one memory-bound candidate.
2. Pick one compute-bound candidate.
3. Pick one suspicious or below-roof candidate.
4. Record CUDA-event timing and derived metrics.
5. If `nsys` is available, collect timeline summaries.
6. Decide whether the profiler evidence supports your bottleneck
   hypothesis.

## Build Target Programs

Build the Week 3 programs first:

```bash
cd ../01_arithmetic_intensity
make

cd ../03_memory_vs_compute
make
```

Then return here:

```bash
cd ../05_performance_counter_workflow
```

## Required Path - Derived Metrics

Run the arithmetic-intensity sweep:

```bash
make run-arithmetic
```

Run the memory-vs-compute comparison:

```bash
make run-memory-compute
```

Fill the first table in `counter_notes_template.md`.

## Recommended Path - Nsight Systems

Check availability:

```bash
nsys --version
```

Profile the arithmetic-intensity sweep:

```bash
make profile-arithmetic-nsys
make stats-arithmetic-nsys
```

Profile the memory-vs-compute comparison:

```bash
make profile-memory-compute-nsys
make stats-memory-compute-nsys
```

Look for:

- CUDA kernel summary
- kernel durations
- host gaps between launches
- unexpected `cudaMemcpy` activity
- synchronization points

## Optional Path - Nsight Compute Compatibility Check

Try this only as a compatibility check:

```bash
ncu --version
make profile-arithmetic-ncu
```

If `ncu` reports unsupported architecture or unsupported device, record that
in the notes and continue with CUDA events plus Nsight Systems.

## Counter Mapping Cheat Sheet

| Diagnosis | Ideal counter signal | Fallback signal on this setup |
| --------- | -------------------- | ----------------------------- |
| Memory-bound | DRAM throughput near roof | Effective bandwidth near measured memory plateau |
| Compute-bound | FP32 pipe utilization near roof | GFLOP/s rises while bandwidth number falls |
| Low occupancy | Occupancy and launch stats | Try block-size sweep and compare runtime |
| Warp stalls | Warp state / scheduler stats | Change access pattern or dependency and compare |
| Launch overhead | Timeline launch/kernel ratio | Nsight Systems kernel summary and host gaps |
| Copy overhead | Memcpy timeline and sizes | Nsight Systems CUDA memory summary |

## Suggested Measurements

| Kernel | Expected bottleneck | CUDA-event metric | Nsight Systems signal | Optional counter signal | Supports hypothesis? |
| ------ | ------------------- | ----------------- | --------------------- | ----------------------- | -------------------- |
| arithmetic_intensity_fma_1 | | | | | |
| arithmetic_intensity_fma_1024 | | | | | |
| memory_stream | | | | | |
| compute_chain_512 | | | | | |
| suspicious_kernel | | | | | |

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


## Connection To Previous Labs

Week 3 built a performance-model ladder:

1. Arithmetic intensity gives the FLOP/byte coordinate.
2. Roofline gives the first bottleneck prediction.
3. Bottleneck analysis turns the prediction into a testable hypothesis.
4. This lab checks that hypothesis with whatever profiling signals are
   available on the local machine.

## Next Experiment

After Week 3, move into GPU simulation:

```text
../../week04
```
