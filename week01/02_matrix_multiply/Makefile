NVCC ?= nvcc
ARCH ?= sm_61
NVCCFLAGS ?= -O3 -std=c++17

TARGETS := matmul_naive

.PHONY: all run clean

all: $(TARGETS)

matmul_naive: matmul_naive.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

run: matmul_naive
	./matmul_naive 1024 16 20

clean:
	rm -f $(TARGETS)
