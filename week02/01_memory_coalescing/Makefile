NVCC ?= nvcc
ARCH ?= sm_61
NVCCFLAGS ?= -O3 -std=c++17

TARGETS := memory_coalescing

.PHONY: all run clean

all: $(TARGETS)

memory_coalescing: memory_coalescing.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

run: memory_coalescing
	./memory_coalescing 16777216 256 100

clean:
	rm -f $(TARGETS)
