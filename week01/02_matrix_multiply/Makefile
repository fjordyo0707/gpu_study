NVCC ?= nvcc
ARCH ?= sm_61
NVCCFLAGS ?= -O3 -std=c++17

TARGETS := \
	matmul_naive \
	matmul_tiled \
	matmul_tile_sweep \
	matmul_register_blocking \
	matmul_memory_layout \
	matmul_cublas

.PHONY: all run run-naive run-tiled run-tile-sweep run-register-blocking \
	run-memory-layout run-cublas clean

all: $(TARGETS)

matmul_naive: matmul_naive.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

matmul_tiled: matmul_tiled.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

matmul_tile_sweep: matmul_tile_sweep.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

matmul_register_blocking: matmul_register_blocking.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

matmul_memory_layout: matmul_memory_layout.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

matmul_cublas: matmul_cublas.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@ -lcublas

run: run-tiled

run-naive: matmul_naive
	./matmul_naive 1024 16 20

run-tiled: matmul_tiled
	./matmul_tiled 1024 16 20

run-tile-sweep: matmul_tile_sweep
	./matmul_tile_sweep 1024 20 8 16 32

run-register-blocking: matmul_register_blocking
	./matmul_register_blocking 1024 16 20

run-memory-layout: matmul_memory_layout
	./matmul_memory_layout 1024 16 20

run-cublas: matmul_cublas
	./matmul_cublas 1024 16 20

clean:
	rm -f $(TARGETS)
