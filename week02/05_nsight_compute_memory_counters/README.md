# Experiment 05 - Profiling Alternatives For Memory Labs

## Objective

Explain the Week 2 memory experiments with timing, derived metrics, and
profiler output that works on the current GTX 1080 Ti / Pascal setup.

This lab was originally planned around Nsight Compute memory counters. On
this machine, that should not be the required path: modern Nsight Compute
removed Pascal `SM 6.x` support, and this repo's GPU is Pascal `sm_61`.

So the lab becomes:

- primary path: CUDA event timing plus derived bandwidth / throughput
- optional path: Nsight Systems timeline profiling
- stretch path: Nsight Compute, `nvprof`, or CUPTI if your local toolchain
  happens to support them

## First Question

Can we still explain the global-memory, cache, and shared-memory behavior
without relying on Nsight Compute hardware counters?

## Hypothesis

Yes. Hardware counters are useful, but they are not the only way to reason
about performance.

For these Week 2 kernels, a careful timing table plus derived metrics should
still reveal:

- whether the kernel is mostly bandwidth-limited
- whether shared-memory staging pays for its extra instructions and barriers
- whether a pattern is sensitive to cache reuse
- whether a shared-memory pattern becomes unusually slow
- whether a profiler timeline agrees with the CUDA-event timings

Nsight Systems will not replace detailed per-kernel memory counters, but it
can still answer useful questions about kernel duration, launch ordering,
host/device synchronization, and memory-copy activity.

## Recommended Reading

- [Nsight Compute Release Notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html):
  check tool support before building a lab around specific profiler
  counters. The 2020.1 notes are the important part for Pascal.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html):
  use this for `nsys profile` and `nsys stats`, which are the practical
  profiling alternatives for this machine.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  review the timing, bandwidth, and memory optimization workflow sections.
- [CUPTI Documentation](https://docs.nvidia.com/cupti/index.html):
  stretch reading only. CUPTI is what many profiling tools build on, but
  writing a CUPTI profiler is too much overhead for this lab.
- [Hierarchical Roofline Analysis: How to Collect Data using Performance Tools on Intel CPUs and NVIDIA GPUs](https://arxiv.org/abs/2009.02449):
  useful for understanding how profiler numbers become architecture-level
  explanations.

## Why Nsight Compute Is Optional Here

The repo hardware is:

```text
NVIDIA GeForce GTX 1080 Ti
Pascal
Compute capability 6.1 / sm_61
```

Nsight Compute release notes say that support for Pascal `SM 6.x` was
removed in Nsight Compute 2020.1. That means a modern CUDA install can have
`ncu` on the system but still fail to profile this GPU.

That is not a learning failure. It is a real profiling constraint, and GPU
engineers run into this kind of tool / architecture mismatch often.

## Implementation TODOs

This lab is a profiling and analysis exercise, so there is no new CUDA
kernel to fill in.

Required work:

1. Build the previous Week 2 executables.
2. Run the primary CUDA-event timing path.
3. Fill in the timing and derived-metric table.
4. If `nsys` is available, collect one timeline report.
5. Explain whether the profiler timeline supports your timing-based
   interpretation.

Optional stretch work:

1. Try `ncu --version` and one short `ncu` run.
2. If `ncu` fails on Pascal, record the error message and move on.
3. Try `nvprof --version` only if it is installed.
4. Read CUPTI docs, but do not implement a CUPTI profiler yet.

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

## Primary Path - CUDA Event Timing

Run the global-vs-shared reuse lab:

```bash
make run-global-shared | tee global_vs_shared_record.log
```

Run the shared-memory bank-conflict lab:

```bash
make run-bank-conflicts | tee shared_bank_conflicts_record.log
```

These runs are the required measurement path for this lab.

## Optional Path - Nsight Systems

Check whether Nsight Systems is available:

```bash
nsys --version
```

Profile the global-vs-shared reuse lab:

```bash
make profile-global-shared-nsys
make stats-global-shared-nsys
```

Profile the shared-memory bank-conflict lab:

```bash
make profile-bank-conflicts-nsys
make stats-bank-conflicts-nsys
```

What to look for in `nsys stats`:

- CUDA kernel summary
- kernel duration agreement with CUDA-event timing
- unexpected host-side gaps between launches
- unexpected CUDA memory copies
- whether one run contains extra synchronization overhead

## Stretch Path - Nsight Compute, nvprof, CUPTI

Try this only as a compatibility check:

```bash
ncu --version
ncu --set detailed ../04_global_vs_shared_reuse/global_vs_shared_reuse 16777216 256 4 32 5
```

If Nsight Compute reports that profiling is unsupported for this GPU, write
that down and stop using it for this course path.

If `nvprof` exists on your system, you can try:

```bash
nvprof ../04_global_vs_shared_reuse/global_vs_shared_reuse 16777216 256 4 32 20
```

Treat `nvprof` as legacy support, not the main lab requirement.

CUPTI is useful background knowledge for later tool-building work, but it is
not required here.

## Suggested Measurements

Fill this table from CUDA-event logs first.

| Lab | Kernel / case | Input size | Repeats / stride | Avg time (ms) | Derived bandwidth or throughput | PASS? |
| --- | ------------- | ---------: | ---------------: | ------------: | ------------------------------: | ----- |
| global_vs_shared_reuse | direct_global | | | | | |
| global_vs_shared_reuse | shared_tiled | | | | | |
| global_vs_shared_reuse | shared_padded | | | | | |
| shared_bank_conflicts | stride 1 | | | | | |
| shared_bank_conflicts | stride 32 | | | | | |

If you run Nsight Systems, fill this second table.

| Lab | Kernel / case | CUDA-event time (ms) | Nsight Systems time (ms) | Host gap visible? | Extra memcpy visible? | Notes |
| --- | ------------- | -------------------: | -----------------------: | ----------------- | -------------------- | ----- |
| global_vs_shared_reuse | direct_global | | | | | |
| global_vs_shared_reuse | shared_tiled | | | | | |
| global_vs_shared_reuse | shared_padded | | | | | |
| shared_bank_conflicts | stride 1 | | | | | |
| shared_bank_conflicts | stride 32 | | | | | |

## Observation Questions

1.
2.
3.

## Observation


## Interpretation


## What This Lab Teaches

Tool choice depends on hardware, driver, CUDA version, and profiler version.

Nsight Compute is the right tool for detailed kernel counters on supported
GPUs. On this Pascal machine, the practical course path is to keep measuring
carefully with CUDA events, use Nsight Systems for timelines when available,
and save full counter-based work for a newer GPU or a compatible profiler
environment.

The important skill is not memorizing one profiler command. The important
skill is building a measurement story that still works when one tool is not
available.

## Connection To Previous Labs

The earlier Week 2 labs answered which versions were faster.

This lab asks how much we can explain using portable measurements:

- Did the timing change match the expected memory-access pattern?
- Did shared memory reduce global-memory traffic enough to pay off?
- Did the bank-conflict benchmark show an obvious slow case?
- Did the profiler timeline agree with the CUDA-event timing?

## Next Experiment

After this profiling-alternatives checkpoint, move into Week 3 performance
modeling:

```text
../../week03/01_arithmetic_intensity
```
