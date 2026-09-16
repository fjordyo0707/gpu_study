NSYS ?= nsys
NCU ?= ncu

GLOBAL_SHARED_BIN ?= ../04_global_vs_shared_reuse/global_vs_shared_reuse
BANK_BIN ?= ../03_shared_memory_bank_conflicts/shared_bank_conflicts

GLOBAL_SHARED_ARGS ?= 16777216 256 4 32 100
BANK_ARGS ?= 16777216 256 256 100

.PHONY: run-global-shared run-bank-conflicts \
	profile-global-shared-nsys profile-bank-conflicts-nsys \
	stats-global-shared-nsys stats-bank-conflicts-nsys \
	profile-global-shared-ncu profile-bank-conflicts-ncu clean

run-global-shared:
	$(GLOBAL_SHARED_BIN) $(GLOBAL_SHARED_ARGS)

run-bank-conflicts:
	$(BANK_BIN) $(BANK_ARGS)

profile-global-shared-nsys:
	$(NSYS) profile --trace=cuda,nvtx,osrt --sample=none \
		-o global_vs_shared_reuse --force-overwrite=true \
		$(GLOBAL_SHARED_BIN) $(GLOBAL_SHARED_ARGS)

profile-bank-conflicts-nsys:
	$(NSYS) profile --trace=cuda,nvtx,osrt --sample=none \
		-o shared_bank_conflicts --force-overwrite=true \
		$(BANK_BIN) $(BANK_ARGS)

stats-global-shared-nsys:
	$(NSYS) stats global_vs_shared_reuse.nsys-rep

stats-bank-conflicts-nsys:
	$(NSYS) stats shared_bank_conflicts.nsys-rep

profile-global-shared-ncu:
	$(NCU) --set detailed --target-processes all \
		--export global_vs_shared_reuse --force-overwrite \
		$(GLOBAL_SHARED_BIN) $(GLOBAL_SHARED_ARGS)

profile-bank-conflicts-ncu:
	$(NCU) --set detailed --target-processes all \
		--export shared_bank_conflicts --force-overwrite \
		$(BANK_BIN) $(BANK_ARGS)

clean:
	rm -f *.nsys-rep *.sqlite *.qdstrm *.ncu-rep
