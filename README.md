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
- [ ] Matrix multiplication tile-size sweep

## Current Focus

Week 1 / Experiment 03:

- Sweep matrix-multiplication tile sizes and block shapes.
- Compare each variant against the naive and shared-memory tiled
  baselines.
- Record how tile size changes runtime, throughput, shared-memory use,
  and occupancy.
