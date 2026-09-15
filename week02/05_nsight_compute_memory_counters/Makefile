NCU ?= ncu

GLOBAL_SHARED_BIN ?= ../04_global_vs_shared_reuse/global_vs_shared_reuse
BANK_BIN ?= ../03_shared_memory_bank_conflicts/shared_bank_conflicts

GLOBAL_SHARED_ARGS ?= 16777216 256 4 32 20
BANK_ARGS ?= 16777216 256 256 20

.PHONY: profile-global-shared profile-bank-conflicts clean

profile-global-shared:
	$(NCU) --set detailed --target-processes all \
		--export global_vs_shared_reuse --force-overwrite \
		$(GLOBAL_SHARED_BIN) $(GLOBAL_SHARED_ARGS)

profile-bank-conflicts:
	$(NCU) --set detailed --target-processes all \
		--export shared_bank_conflicts --force-overwrite \
		$(BANK_BIN) $(BANK_ARGS)

clean:
	rm -f *.ncu-rep
