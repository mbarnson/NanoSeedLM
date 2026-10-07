# NanoSeedLM: SeedLM-compressed MoE experts for K2-Horizon-MoVA-36B-A4B on Apple Silicon (Metal).
#   make / make build   metallibs into out/res, binaries into out/bin
#   make test           every tests/test_* binary (test_affine's MLX goldens come from $(PY) tools/mova_affine_golden.py)
#   make clean

OUT       := out
BIN       := $(OUT)/bin
RES       := $(OUT)/res
OBJ       := $(OUT)/obj
CC        := clang
CWARN     := -Wall -Wextra -Wno-unused-function -Wno-unused-parameter
OBJCFLAGS := -O2 -g -fobjc-arc $(CWARN) -Iengine -Iharness -Inslm
# no FMA contraction: scalar, vector and GPU paths must agree bit for bit
CFLAGS    := -std=c99 -O2 -g $(CWARN) -ffp-contract=off -Inslm -Iharness
LIBS      := -framework Metal -framework Foundation -framework CoreFoundation -framework IOKit
METAL     := xcrun -sdk macosx metal -O3 -std=metal3.2
PY        ?= python3

NSLM_LIB  := $(wildcard nslm/lib_*.c)
NSLM_HDRS := $(wildcard nslm/*.h)
ENGINE    := engine/mova_gpu.m nslm/lib_model_st.c nslm/lib_json.c nslm/lib_mova_cfg.c nslm/lib_mova_ckpt.c nslm/lib_moe.c nslm/lib_format.c nslm/lib_sha256.c
SERVE_C   := harness/chat_template.c harness/tool_calls.c
ENG_HDRS  := engine/engine_api.h engine/mova_ext.h engine/kernels_moe.metal nslm/model_st.h nslm/json.h nslm/mova_cfg.h nslm/mova_ckpt.h nslm/lfsr.h

METALLIBS := $(RES)/kernels_moe.metallib $(RES)/search.metallib $(RES)/search4.metallib
ENG_TOOLS := nslm-chat nslm-mova-smoke nslm-mova-gen nslm-mova-score nslm-mova-plcheck
TOOLS     := $(addprefix $(BIN)/,$(ENG_TOOLS) nslm-serve nslm-mova-bench nslm-mova-kbench nslm-moe nslm-mova-pack nslm-bits-probe)
TESTS     := $(patsubst tests/%.c,$(BIN)/%,$(wildcard tests/test_*.c)) $(patsubst tests/%.m,$(BIN)/%,$(wildcard tests/test_*.m))

.PHONY: all build test test-mlx clean
all: build
build: $(METALLIBS) $(TOOLS)

# ---- Metal libraries (the search kernels without fast math: their f32 arithmetic must equal the CPU's) -----------
$(RES)/kernels_moe.metallib: engine/kernels_moe.metal
	@mkdir -p $(@D)
	$(METAL) -c $< -o $(@D)/kernels_moe.air && xcrun -sdk macosx metallib $(@D)/kernels_moe.air -o $@
$(RES)/search.metallib: nslm/search.metal nslm/search_gpu.h nslm/lfsr.h
	@mkdir -p $(@D)
	$(METAL) -fno-fast-math -Inslm -c $< -o $(@D)/search.air && xcrun -sdk macosx metallib $(@D)/search.air -o $@
$(RES)/search4.metallib: nslm/search4.metal nslm/search4_gpu.h nslm/lfsr.h
	@mkdir -p $(@D)
	$(METAL) -fno-fast-math -Inslm -c $< -o $(@D)/search4.air && xcrun -sdk macosx metallib $(@D)/search4.air -o $@

# ---- engine tools ---------------------------------------------------------------------------------------------------
$(addprefix $(BIN)/,$(ENG_TOOLS)): $(BIN)/%: harness/%.m harness/tokenizer.m harness/tokenizer.h $(ENGINE) $(ENG_HDRS)
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) $< harness/tokenizer.m $(ENGINE) $(LIBS) -o $@
SERVE_O   := $(patsubst harness/%.c,$(OBJ)/%.o,$(SERVE_C))
$(OBJ)/%.o: harness/%.c nslm/json.h harness/chat_template.h harness/tool_calls.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
$(BIN)/nslm-serve: harness/nslm-serve.m harness/tokenizer.m harness/tokenizer.h $(SERVE_O) $(ENGINE) $(ENG_HDRS)
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) harness/nslm-serve.m harness/tokenizer.m $(SERVE_O) $(ENGINE) $(LIBS) -o $@
$(BIN)/nslm-mova-bench: harness/nslm-mova-bench.m harness/telemetry.m harness/telemetry.h harness/common.h $(ENGINE) $(ENG_HDRS)
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) harness/nslm-mova-bench.m harness/telemetry.m $(ENGINE) $(LIBS) -o $@
$(BIN)/nslm-mova-kbench: harness/nslm-mova-kbench.m harness/common.h engine/kernels_moe.metal nslm/lfsr.h
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) harness/nslm-mova-kbench.m $(LIBS) -o $@

# ---- compressor: seed search (C99 + the Metal search glue) and the packer -----------------------------------------
$(BIN)/nslm-moe: nslm/moe.c nslm/search_metal.m nslm/search4_metal.m $(NSLM_HDRS) $(NSLM_LIB)
	@mkdir -p $(@D) $(OBJ)
	$(CC) $(CFLAGS) -O3 -c nslm/moe.c -o $(OBJ)/moe.o
	$(CC) $(OBJCFLAGS) -c nslm/search_metal.m -o $(OBJ)/search_metal.o
	$(CC) $(OBJCFLAGS) -c nslm/search4_metal.m -o $(OBJ)/search4_metal.o
	$(CC) $(CFLAGS) -O3 $(OBJ)/moe.o $(OBJ)/search_metal.o $(OBJ)/search4_metal.o $(NSLM_LIB) -o $@ -lm -lpthread $(LIBS)
$(BIN)/nslm-mova-pack: nslm/mova_pack.c $(NSLM_HDRS) $(NSLM_LIB)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -O3 $< $(NSLM_LIB) -o $@ -lm -lpthread
$(BIN)/nslm-bits-probe: nslm/bits_probe.c $(NSLM_HDRS) $(NSLM_LIB)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -O3 $< $(NSLM_LIB) -o $@ -lm -lpthread

# ---- tests ----------------------------------------------------------------------------------------------------------
$(BIN)/test_%: tests/test_%.c $(NSLM_HDRS) $(NSLM_LIB)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(NSLM_LIB) -o $@ -lm -lpthread
$(BIN)/test_chat_template $(BIN)/test_tool_calls: $(BIN)/test_%: tests/test_%.c $(SERVE_C) $(SERVE_C:.c=.h) nslm/lib_json.c tests/data/template_golden.json
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -Iharness $< $(SERVE_C) nslm/lib_json.c -o $@ -lm
$(BIN)/test_serve_splitter: tests/test_serve_splitter.m harness/nslm-serve.m harness/tokenizer.m harness/tokenizer.h $(SERVE_O) $(ENGINE) $(ENG_HDRS)
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) tests/test_serve_splitter.m harness/tokenizer.m $(SERVE_O) $(ENGINE) $(LIBS) -o $@
$(BIN)/test_search_gpu: tests/test_search_gpu.m $(NSLM_HDRS) $(NSLM_LIB) $(RES)/search.metallib
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) -ffp-contract=off tests/test_search_gpu.m $(NSLM_LIB) $(LIBS) -o $@
$(BIN)/test_search4_gpu: tests/test_search4_gpu.m nslm/search4_metal.m $(NSLM_HDRS) $(NSLM_LIB) $(RES)/search4.metallib
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) -ffp-contract=off tests/test_search4_gpu.m nslm/search4_metal.m $(NSLM_LIB) $(LIBS) -o $@
$(BIN)/test_mova_kernels: tests/test_mova_kernels.m engine/kernels_moe.metal nslm/lfsr.h nslm/lib_affine.c $(RES)/kernels_moe.metallib
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) tests/test_mova_kernels.m nslm/lib_affine.c $(LIBS) -o $@

$(OUT)/test/affine/index.txt: tools/mova_affine_golden.py tools/mova_common.py
	$(PY) tools/mova_affine_golden.py --out $(OUT)/test/affine

# the MLX loader (tools/nanoseedlm_k2.py); OMLX_K2_MODEL = oMLX's k2_horizon_model.py
test-mlx:
	$(PY) tools/test_nanoseedlm_k2.py

test: $(TESTS) $(OUT)/test/affine/index.txt
	@for t in $(TESTS); do echo "== $$t"; $$t || exit 1; done; echo "all tests passed"

clean:
	rm -rf $(OUT)
