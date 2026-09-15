NVCC ?= nvcc
ARCH ?= sm_61
NVCCFLAGS ?= -O3 -std=c++17

TARGETS := shared_bank_conflicts

.PHONY: all run clean

all: $(TARGETS)

shared_bank_conflicts: shared_bank_conflicts.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

run: shared_bank_conflicts
	./shared_bank_conflicts 16777216 256 256 100

clean:
	rm -f $(TARGETS)
