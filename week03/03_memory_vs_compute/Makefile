NVCC ?= nvcc
ARCH ?= sm_61
NVCCFLAGS ?= -O3 -std=c++17

TARGETS := memory_vs_compute

.PHONY: all run clean

all: $(TARGETS)

memory_vs_compute: memory_vs_compute.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

run: memory_vs_compute
	./memory_vs_compute 16777216 256 100

clean:
	rm -f $(TARGETS)
