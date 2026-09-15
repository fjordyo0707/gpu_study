NVCC ?= nvcc
ARCH ?= sm_61
NVCCFLAGS ?= -O3 -std=c++17

TARGETS := arithmetic_intensity

.PHONY: all run clean

all: $(TARGETS)

arithmetic_intensity: arithmetic_intensity.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

run: arithmetic_intensity
	./arithmetic_intensity 16777216 256 100

clean:
	rm -f $(TARGETS)
