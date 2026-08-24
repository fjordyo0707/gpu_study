NVCC ?= nvcc
ARCH ?= sm_61
NVCCFLAGS ?= -O3 -std=c++17

TARGETS := cache_behavior

.PHONY: all run clean

all: $(TARGETS)

cache_behavior: cache_behavior.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

run: cache_behavior
	./cache_behavior 1048576 256 64 100

clean:
	rm -f $(TARGETS)
