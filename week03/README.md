# Week 3 - Performance Modeling

Week 3 turns raw benchmark numbers into a performance model.

The goal is to learn how to answer:

```text
Is this kernel limited by memory bandwidth, compute throughput, launch/copy
overhead, or something else?
```

## Labs

| Experiment | Lab | Main idea | Status |
| ---------- | --- | --------- | ------ |
| 01 | `01_arithmetic_intensity` | Increase FLOPs per byte and measure the transition from memory-like to compute-like behavior | Completed once |
| 02 | `02_roofline_model` | Plot kernels against memory and compute roofs | Starter analysis lab |
| 03 | `03_memory_vs_compute` | Build direct memory-bound, compute-bound, and mixed kernels | Starter CUDA lab |
| 04 | `04_bottleneck_analysis` | Turn measurements into a bottleneck hypothesis and next experiment | Starter analysis lab |
| 05 | `05_performance_counter_workflow` | Compare model predictions with profiler/timeline signals available on this machine | Starter workflow lab |

## Recommended Order

1. Finish `02_roofline_model` using results from Week 1, Week 2, and
   `01_arithmetic_intensity`.
2. Implement the TODO kernels in `03_memory_vs_compute`.
3. Add those results back into the Roofline model.
4. Use `04_bottleneck_analysis` to write hypotheses for slow kernels.
5. Use `05_performance_counter_workflow` to check the hypotheses with CUDA
   event timing, Nsight Systems, and optional profiler tools.

## Notes For This GPU

The repo uses a GTX 1080 Ti / Pascal GPU with compute capability `sm_61`.
Modern Nsight Compute may not support detailed counter collection on this
device, so Week 3 keeps CUDA-event timing and derived metrics as the
required path.

Nsight Systems is still useful when available because it can show kernel
duration, host gaps, copies, and synchronization behavior.
