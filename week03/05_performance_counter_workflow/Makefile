NSYS ?= nsys
NCU ?= ncu

ARITH_BIN ?= ../01_arithmetic_intensity/arithmetic_intensity
MEMORY_COMPUTE_BIN ?= ../03_memory_vs_compute/memory_vs_compute

ARITH_ARGS ?= 16777216 256 100
MEMORY_COMPUTE_ARGS ?= 16777216 256 100

.PHONY: run-arithmetic run-memory-compute \
	profile-arithmetic-nsys profile-memory-compute-nsys \
	stats-arithmetic-nsys stats-memory-compute-nsys \
	profile-arithmetic-ncu profile-memory-compute-ncu clean

run-arithmetic:
	$(ARITH_BIN) $(ARITH_ARGS)

run-memory-compute:
	$(MEMORY_COMPUTE_BIN) $(MEMORY_COMPUTE_ARGS)

profile-arithmetic-nsys:
	$(NSYS) profile --trace=cuda,nvtx,osrt --sample=none \
		-o arithmetic_intensity --force-overwrite=true \
		$(ARITH_BIN) $(ARITH_ARGS)

profile-memory-compute-nsys:
	$(NSYS) profile --trace=cuda,nvtx,osrt --sample=none \
		-o memory_vs_compute --force-overwrite=true \
		$(MEMORY_COMPUTE_BIN) $(MEMORY_COMPUTE_ARGS)

stats-arithmetic-nsys:
	$(NSYS) stats arithmetic_intensity.nsys-rep

stats-memory-compute-nsys:
	$(NSYS) stats memory_vs_compute.nsys-rep

profile-arithmetic-ncu:
	$(NCU) --set detailed --target-processes all \
		--export arithmetic_intensity --force-overwrite \
		$(ARITH_BIN) $(ARITH_ARGS)

profile-memory-compute-ncu:
	$(NCU) --set detailed --target-processes all \
		--export memory_vs_compute --force-overwrite \
		$(MEMORY_COMPUTE_BIN) $(MEMORY_COMPUTE_ARGS)

clean:
	rm -f *.nsys-rep *.sqlite *.qdstrm *.ncu-rep
