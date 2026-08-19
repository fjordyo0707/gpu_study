NVCC ?= nvcc
ARCH ?= sm_61
NVCCFLAGS ?= -O3 -std=c++17

TARGETS := matmul_naive matmul_tiled

.PHONY: all run run-naive run-tiled clean

all: $(TARGETS)

matmul_naive: matmul_naive.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

matmul_tiled: matmul_tiled.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

run: run-tiled

run-naive: matmul_naive
	./matmul_naive 1024 16 20

run-tiled: matmul_tiled
	./matmul_tiled 1024 16 20

clean:
	rm -f $(TARGETS)
