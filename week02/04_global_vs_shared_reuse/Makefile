NVCC ?= nvcc
ARCH ?= sm_61
NVCCFLAGS ?= -O3 -std=c++17

TARGETS := global_vs_shared_reuse

.PHONY: all run clean

all: $(TARGETS)

global_vs_shared_reuse: global_vs_shared_reuse.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

run: global_vs_shared_reuse
	./global_vs_shared_reuse 16777216 256 4 32 100

clean:
	rm -f $(TARGETS)
