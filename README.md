# GPU Study

A hands-on GPU engineering study focused on understanding GPU
architecture, performance, compilers, and open-source GPU software.

The goal is not simply to learn CUDA APIs, but to build the ability to:

- design GPU experiments
- measure GPU performance
- understand GPU microarchitecture
- analyze memory behavior
- understand GPU scheduling
- understand GPU compilation
- work with GPU simulators
- read and modify GPU compiler/driver code
- contribute to open-source GPU projects

## Hardware

- GPU: NVIDIA GeForce GTX 1080 Ti
- Architecture: Pascal
- Compute Capability: 6.1 (`sm_61`)
- VRAM: 11 GiB
- Theoretical memory bandwidth: ~484 GB/s

## Software Environment

- OS: Ubuntu 24.04 LTS
- Kernel: 6.17.0-19-generic
- NVIDIA Driver: 580.173.02
- CUDA Toolkit: 12.6.3
- `nvcc`: 12.6.85
- GCC: 13.3.0

## Study Roadmap

### Week 1 — GPU Performance Fundamentals

- CUDA programming basics
- GPU threads and blocks
- Global memory
- Kernel timing
- Memory bandwidth
- Vector addition
- Matrix multiplication
- Performance measurement

### Week 2 — GPU Memory Hierarchy

- Memory coalescing
- Global memory
- L2 cache
- Shared memory
- Cache behavior
- Memory access patterns

### Week 3 — Performance Modeling

- Arithmetic intensity
- Memory-bound vs compute-bound workloads
- Roofline model
- Bottleneck analysis
- GPU performance counters

### Week 4 — GPU Simulation

- Accel-Sim
- GPU workload tracing
- Simulation configuration
- Comparing simulation and real hardware

### Week 5 — GPU Microarchitecture

- SM architecture
- Warps
- Warp scheduling
- Instruction issue
- Registers
- Occupancy
- Latency hiding

### Week 6 — GPU Scheduling

- Warp schedulers
- Scheduling policies
- Experiments with scheduling behavior
- Performance impact

### Week 7 — LLVM / MLIR

- LLVM IR
- MLIR
- GPU dialect
- NVVM
- GPU compilation pipeline

### Week 8 — GPU Compiler

- Compilation pipeline
- Optimization passes
- Register usage
- Instruction selection
- Code generation

### Week 9 — Compiler → Hardware

- Compiler optimization vs GPU performance
- Register pressure
- Occupancy
- Instruction-level behavior
- Performance analysis

### Week 10 — Mesa / RADV

- Vulkan driver architecture
- Shader compilation
- RADV
- NIR
- AMD GPU compiler concepts

### Week 11 — Open Source

- Select an open-source GPU project
- Build from source
- Find a small issue
- Make a contribution
- Submit a pull request

### Week 12 — Capstone

Build a complete GPU engineering project combining:

- GPU performance analysis
- architecture
- compiler or simulator
- experiments
- documentation

## Experiment Methodology

Every experiment should follow:

1. Ask a question
2. Form a hypothesis
3. Design an experiment
4. Control variables
5. Measure
6. Analyze results
7. Change one thing
8. Measure again
9. Explain the result

The goal is to understand **why** the GPU behaves the way it does.

## Recommended Reading Convention

Every practice lab README should include a `## Recommended Reading`
section near the top.

Keep the list short and useful:

- Prefer official documentation for CUDA behavior and tools.
- Add NVIDIA technical blogs or classic papers when they explain the
  experiment especially well.
- Include one sentence explaining why each reading matters for the lab.
- Keep optional/stretch material clearly marked so it does not distract
  from the current exercise.

Core references for Week 1 and Week 2:

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html):
  the main reference for CUDA's programming model, kernels, thread
  hierarchy, memory spaces, and execution behavior.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  the practical performance guide for timing, bandwidth, coalescing,
  shared memory, occupancy, and optimization workflow.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html):
  the profiler reference to use when supported hardware is available and a
  benchmark number needs to be explained with memory and compute counters.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html):
  the timeline profiler reference to use when detailed Nsight Compute
  counters are unavailable.

Research paper track:

- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785)
  by Williams, Waterman, and Patterson: the classic, readable bridge from
  benchmark numbers to computer-architecture limits.
- [Hierarchical Roofline Analysis: How to Collect Data using Performance Tools on Intel CPUs and NVIDIA GPUs](https://arxiv.org/abs/2009.02449)
  by Yang: a practical follow-up for turning profiler counters into
  cache-level performance explanations.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
  by Dao et al.: a modern, approachable example of designing an algorithm
  around GPU memory hierarchy instead of FLOP count alone.
- [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis](https://arxiv.org/abs/2501.12084)
  by Luo et al.: recent GPU architecture research; read the abstract,
  introduction, memory-subsystem parts, and figures first.
- [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/abs/2507.10789)
  by Jarmusch, Graddon, and Chandrasekaran: latest stretch reading; focus
  on methodology and architecture patterns rather than exact numbers for
  your GTX 1080 Ti.

## Current Progress

- [x] Ubuntu + GPU environment
- [x] NVIDIA driver
- [x] Secure Boot / MOK configuration
- [x] CUDA Toolkit
- [x] First CUDA kernel
- [x] Vector addition correctness
- [x] Initial performance measurement
- [x] Proper microbenchmark
- [x] Memory bandwidth analysis
- [x] Memory access experiments
- [x] Matrix multiplication
- [x] Shared-memory tiled matrix multiplication
- [x] Matrix multiplication tile-size sweep
- [x] Matrix multiplication register blocking
- [x] Matrix multiplication memory-layout experiment
- [x] Matrix multiplication cuBLAS comparison
- [x] Warp-level memory coalescing
- [x] Cache behavior / working-set size sweep
- [x] Shared-memory bank conflicts
- [ ] Global memory reuse vs shared-memory reuse
- [ ] Profiling alternatives for memory labs
- [ ] Arithmetic intensity sweep
- [ ] Simple Roofline model

## Current Focus

Week 2 / Experiment 04:

- Compare repeated direct global-memory reads against explicit
  shared-memory staging.
- Measure when shared memory helps and when coalesced global loads plus
  hardware cache are already good enough.
- Connect the result back to matrix tiling, cache reuse, and shared-memory
  overhead.

## Upcoming Practice Queue

1. Week 2 / Experiment 05 - Profiling alternatives for memory labs
2. Week 3 / Experiment 01 - Arithmetic intensity
3. Week 3 / Experiment 02 - Roofline model
