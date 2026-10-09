# NanoSeedLM on macOS / Apple Silicon (Metal).  Windows and Linux with NVIDIA CUDA: CMakeLists.txt (see README).
#   make / make build   metallibs into out/res, binaries into out/bin
#   make test           every test (test_affine's MLX goldens come from $(PY) tools/mova_affine_golden.py)
#   make clean
#
# The command-line tools, the tokenizer and the tests are C, shared with the CUDA build; Objective-C remains only for
# the Metal engine (engine/mova_gpu.m), the Metal search glue, and the Metal-specific kernel benchmark and kernel test.

OUT       := out
BIN       := $(OUT)/bin
RES       := $(OUT)/res
OBJ       := $(OUT)/obj
CC        := clang
CWARN     := -Wall -Wextra -Wno-unused-function -Wno-unused-parameter
OBJCFLAGS := -O2 -g -fobjc-arc $(CWARN) -Iengine -Iharness -Inslm
# no FMA contraction: scalar, vector and GPU paths must agree bit for bit
CFLAGS    := -std=c11 -O2 -g $(CWARN) -ffp-contract=off -Inslm -Iharness -Iengine
LIBS      := -framework Metal -framework Foundation -framework CoreFoundation -framework IOKit
HLIBS     := -licucore -framework CoreFoundation -framework IOKit
METAL     := xcrun -sdk macosx metal -O3 -std=metal3.2
PY        ?= python3

NSLM_LIB  := $(wildcard nslm/lib_*.c)
NSLM_HDRS := $(wildcard nslm/*.h)
ENGINE    := engine/mova_gpu.m nslm/lib_model_st.c nslm/lib_json.c nslm/lib_mova_cfg.c nslm/lib_mova_ckpt.c nslm/lib_moe.c nslm/lib_format.c nslm/lib_sha256.c
HARNESS   := harness/platform.c harness/tokenizer.c harness/chat_template.c harness/tool_calls.c harness/kv_disk.c
HARN_HDRS := $(wildcard harness/*.h)
ENG_HDRS  := engine/engine_api.h engine/mova_ext.h engine/kernels_moe.metal nslm/model_st.h nslm/json.h nslm/mova_cfg.h nslm/mova_ckpt.h nslm/lfsr.h

METALLIBS := $(RES)/kernels_moe.metallib $(RES)/search.metallib $(RES)/search4.metallib $(RES)/searchp.metallib
ENG_TOOLS := nslm-chat nslm-serve nslm-mova-smoke nslm-mova-gen nslm-mova-score nslm-mova-plcheck nslm-mova-refcheck nslm-mova-routes \
             nslm-mova-mlacapture
TOOLS     := $(addprefix $(BIN)/,$(ENG_TOOLS) nslm-mova-bench nslm-mova-kbench nslm-moe nslm-dense nslm-mova-pack nslm-bits-probe)
C_TESTS   := $(filter-out tests/test_engine.c tests/test_serve_splitter.c tests/test_search_gpu.c tests/test_search4_gpu.c tests/test_searchp_gpu.c \
                            tests/test_mova_kernels.c,$(wildcard tests/test_*.c))
TESTS     := $(patsubst tests/%.c,$(BIN)/%,$(C_TESTS)) $(BIN)/test_engine $(BIN)/test_serve_splitter $(BIN)/test_search_gpu \
             $(BIN)/test_search4_gpu $(BIN)/test_searchp_gpu $(BIN)/test_mova_kernels

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
$(RES)/searchp.metallib: nslm/searchp.metal nslm/searchp_gpu.h nslm/lfsr.h
	@mkdir -p $(@D)
	$(METAL) -fno-fast-math -Inslm -c $< -o $(@D)/searchp.air && xcrun -sdk macosx metallib $(@D)/searchp.air -o $@

# ---- objects: the engine (Metal), the shared harness and the C library ---------------------------------------------
ENG_O     := $(OBJ)/mova_gpu.o
$(ENG_O): engine/mova_gpu.m $(ENG_HDRS)
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) -c $< -o $@
HARN_O    := $(patsubst harness/%.c,$(OBJ)/h_%.o,$(HARNESS))
$(OBJ)/h_%.o: harness/%.c $(HARN_HDRS) nslm/json.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
LIB_O     := $(patsubst nslm/%.c,$(OBJ)/n_%.o,$(NSLM_LIB))
$(OBJ)/n_%.o: nslm/%.c $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -O3 -c $< -o $@
TELEM_O   := $(OBJ)/telemetry.o
$(TELEM_O): harness/telemetry.m harness/telemetry.h
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) -c $< -o $@

# ---- tools ----------------------------------------------------------------------------------------------------------
$(addprefix $(BIN)/,$(ENG_TOOLS)): $(BIN)/%: harness/%.c $(ENG_O) $(HARN_O) $(LIB_O) $(HARN_HDRS) $(ENG_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(ENG_O) $(HARN_O) $(LIB_O) $(LIBS) $(HLIBS) -lpthread -o $@
$(BIN)/nslm-mova-bench: harness/nslm-mova-bench.c $(ENG_O) $(HARN_O) $(LIB_O) $(TELEM_O) $(HARN_HDRS) $(ENG_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(ENG_O) $(HARN_O) $(LIB_O) $(TELEM_O) $(LIBS) $(HLIBS) -o $@
$(BIN)/nslm-mova-kbench: harness/nslm-mova-kbench.m harness/common.h engine/kernels_moe.metal nslm/lfsr.h
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) harness/nslm-mova-kbench.m $(LIBS) -o $@

# ---- compressor: seed search (C + the Metal search glue) and the packer -------------------------------------------
SEARCH_O  := $(OBJ)/search_metal.o $(OBJ)/search4_metal.o $(OBJ)/searchp_metal.o
$(OBJ)/%_metal.o: nslm/%_metal.m $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) -c $< -o $@
$(BIN)/nslm-moe: nslm/moe.c $(SEARCH_O) $(LIB_O) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -O3 nslm/moe.c $(SEARCH_O) $(LIB_O) -o $@ -lm -lpthread $(LIBS)
$(BIN)/nslm-dense: nslm/dense.c $(OBJ)/search4_metal.o $(OBJ)/searchp_metal.o $(LIB_O) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -O3 nslm/dense.c $(OBJ)/search4_metal.o $(OBJ)/searchp_metal.o $(LIB_O) -o $@ -lm -lpthread $(LIBS)
$(BIN)/nslm-mova-pack: nslm/mova_pack.c $(LIB_O) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -O3 $< $(LIB_O) -o $@ -lm -lpthread
$(BIN)/nslm-bits-probe: nslm/bits_probe.c $(LIB_O) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -O3 $< $(LIB_O) -o $@ -lm -lpthread

# ---- tests (the same C sources as the CUDA build; test_mova_kernels runs on tests/kernel_backend_metal.m) ---------
$(BIN)/test_%: tests/test_%.c $(LIB_O) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(LIB_O) -o $@ -lm -lpthread
$(BIN)/test_chat_template $(BIN)/test_tool_calls $(BIN)/test_tokenizer $(BIN)/test_kv_disk: $(BIN)/test_%: tests/test_%.c $(HARN_O) $(LIB_O) tests/data/template_golden.json
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(HARN_O) $(LIB_O) $(HLIBS) -o $@ -lm
$(BIN)/test_engine: tests/test_engine.c $(ENG_O) $(LIB_O) $(ENG_HDRS) $(RES)/kernels_moe.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(ENG_O) $(LIB_O) $(LIBS) -lpthread -o $@
$(BIN)/test_serve_splitter: tests/test_serve_splitter.c harness/nslm-serve.c $(ENG_O) $(HARN_O) $(LIB_O) $(HARN_HDRS) $(ENG_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(ENG_O) $(HARN_O) $(LIB_O) $(LIBS) $(HLIBS) -lpthread -o $@
$(BIN)/test_search_gpu: tests/test_search_gpu.c $(OBJ)/search_metal.o $(LIB_O) $(NSLM_HDRS) $(RES)/search.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(OBJ)/search_metal.o $(LIB_O) $(LIBS) -lpthread -o $@
$(BIN)/test_searchp_gpu: tests/test_searchp_gpu.c $(OBJ)/searchp_metal.o $(LIB_O) $(NSLM_HDRS) $(RES)/searchp.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(OBJ)/searchp_metal.o $(LIB_O) $(LIBS) -lpthread -o $@
$(BIN)/test_search4_gpu: tests/test_search4_gpu.c $(OBJ)/search4_metal.o $(LIB_O) $(NSLM_HDRS) $(RES)/search4.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(OBJ)/search4_metal.o $(LIB_O) $(LIBS) -lpthread -o $@
$(BIN)/test_mova_kernels: tests/test_mova_kernels.c tests/kernel_backend_metal.m tests/kernel_backend.h engine/kernels_moe.metal \
                          $(LIB_O) $(RES)/kernels_moe.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c tests/test_mova_kernels.c -o $(OBJ)/test_mova_kernels.o
	$(CC) $(OBJCFLAGS) -Itests tests/kernel_backend_metal.m $(OBJ)/test_mova_kernels.o $(LIB_O) $(LIBS) -o $@

$(OUT)/test/affine/index.txt: tools/mova_affine_golden.py tools/mova_common.py
	$(PY) tools/mova_affine_golden.py --out $(OUT)/test/affine

# the MLX loader (tools/nanoseedlm_k2.py); OMLX_K2_MODEL = oMLX's k2_horizon_model.py
test-mlx:
	$(PY) tools/test_nanoseedlm_k2.py

# the MLX goldens of test_affine are built first (macOS has MLX); test_mova_cfg and test_tokenizer read MOVA_DIR (a model
# folder) and are skipped without one.  A test that exits 77 is skipped.
test: $(TESTS) $(OUT)/test/affine/index.txt
	@for t in $(TESTS); do echo "== $$t"; $$t; r=$$?; if [ $$r -eq 77 ]; then echo "(skipped)"; elif [ $$r -ne 0 ]; then exit 1; fi; done; echo "all tests passed"

clean:
	rm -rf $(OUT)
