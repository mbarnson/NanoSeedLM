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
HARNESS   := harness/platform.c harness/tokenizer.c harness/chat_template.c harness/chat_qwen.c harness/chat_model.c harness/tool_calls.c harness/kv_disk.c
HARN_HDRS := $(wildcard harness/*.h)
ENG_HDRS  := engine/engine_api.h engine/mova_ext.h engine/kernels_moe.metal nslm/model_st.h nslm/json.h nslm/mova_cfg.h nslm/mova_ckpt.h nslm/lfsr.h \
             engine/engine_mova.h engine/engine_qwen4.h engine/engine_api_fns.h engine/engine_unname.h engine/qwen4_eng.h

METALLIBS := $(RES)/kernels_moe.metallib $(RES)/kernels_qwen4.metallib $(RES)/search.metallib $(RES)/search4.metallib \
             $(RES)/searchp.metallib $(RES)/trellis.metallib
ENG_TOOLS := nslm-chat nslm-serve nslm-mova-smoke nslm-mova-gen nslm-mova-score nslm-mova-plcheck nslm-mova-refcheck nslm-mova-routes \
             nslm-mova-mlacapture
TOOLS     := $(addprefix $(BIN)/,$(ENG_TOOLS) nslm-mova-bench nslm-mova-kbench nslm-moe nslm-dense nslm-mova-pack nslm-qwen4-pack nslm-qwen4-emulate nslm-bits-probe nslm-cpu-seed-bench \
                             nslm-qwen4-gen)
C_TESTS   := $(filter-out tests/test_engine.c tests/test_engine_formats.c tests/test_serve_splitter.c tests/test_search_gpu.c tests/test_search4_gpu.c tests/test_searchc_gpu.c tests/test_searchp_gpu.c tests/test_trellis_gpu.c tests/test_tq_mv_gpu.c \
                            tests/test_mova_kernels.c tests/test_qwen4_kernels.c \
                            tests/test_qwen4_engine.c tests/test_qwen4_api.c tests/test_qwen4_mtp.c tests/test_qwen4_api_mtp.c tests/test_engine_dispatch.c tests/test_qwen4_metal.c,$(wildcard tests/test_*.c))
TESTS     := $(patsubst tests/%.c,$(BIN)/%,$(C_TESTS)) $(BIN)/test_engine $(BIN)/test_serve_splitter $(BIN)/test_search_gpu \
             $(BIN)/test_search4_gpu $(BIN)/test_searchp_gpu $(BIN)/test_mova_kernels $(BIN)/test_qwen4_kernels $(BIN)/test_qwen4_metal \
             $(BIN)/test_qwen4_mtp $(BIN)/test_qwen4_api_mtp $(BIN)/test_trellis_gpu

.PHONY: all build test test-mlx clean
all: build
build: $(METALLIBS) $(TOOLS)

# ---- Metal libraries (the search kernels without fast math: their f32 arithmetic must equal the CPU's) -----------
$(RES)/kernels_moe.metallib: engine/kernels_moe.metal
	@mkdir -p $(@D)
	$(METAL) -c $< -o $(@D)/kernels_moe.air && xcrun -sdk macosx metallib $(@D)/kernels_moe.air -o $@
# the qwen4 kernels without fast math: they match nslm/qwen4_ops.h within 1e-5 (RoPE at large positions, exp, rsqrt)
$(RES)/kernels_qwen4.metallib: engine/kernels_qwen4.metal
	@mkdir -p $(@D)
	$(METAL) -fno-fast-math -c $< -o $(@D)/kernels_qwen4.air && xcrun -sdk macosx metallib $(@D)/kernels_qwen4.air -o $@
$(RES)/search.metallib: nslm/search.metal nslm/search_gpu.h nslm/lfsr.h
	@mkdir -p $(@D)
	$(METAL) -fno-fast-math -Inslm -c $< -o $(@D)/search.air && xcrun -sdk macosx metallib $(@D)/search.air -o $@
$(RES)/search4.metallib: nslm/search4.metal nslm/search4_gpu.h nslm/lfsr.h
	@mkdir -p $(@D)
	$(METAL) -fno-fast-math -Inslm -c $< -o $(@D)/search4.air && xcrun -sdk macosx metallib $(@D)/search4.air -o $@
$(RES)/searchp.metallib: nslm/searchp.metal nslm/searchp_gpu.h nslm/lfsr.h
	@mkdir -p $(@D)
	$(METAL) -fno-fast-math -Inslm -c $< -o $(@D)/searchp.air && xcrun -sdk macosx metallib $(@D)/searchp.air -o $@
# the trellis encoder's Viterbi: its costs must equal nslm/lib_trellis.c's, so no fast math (and no contraction)
$(RES)/trellis.metallib: nslm/trellis.metal
	@mkdir -p $(@D)
	$(METAL) -fno-fast-math -Inslm -c $< -o $(@D)/trellis.air && xcrun -sdk macosx metallib $(@D)/trellis.air -o $@

# ---- objects: the engine (Metal), the shared harness and the C library ---------------------------------------------
# engine_api.h for every model (option A): engine/engine.c picks the MoVA engine (engine/mova_gpu.m, its eng_* named
# mova_eng_* by engine_mova.h) or the qwen4 one (engine/qwen4_api.c over engine/qwen4_gpu.m, q4_eng_*) by config.json
MOVA_O    := $(OBJ)/mova_gpu.o
$(MOVA_O): engine/mova_gpu.m $(ENG_HDRS)
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) -c $< -o $@
# the qwen4 Metal engine (engine/qwen4_eng.h): alone for nslm-qwen4-gen and test_qwen4_metal, else under qwen4_api.c
Q4E_O     := $(OBJ)/qwen4_gpu.o
$(Q4E_O): engine/qwen4_gpu.m engine/qwen4_eng.h engine/kernels_moe.metal engine/kernels_qwen4.metal $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) -c $< -o $@
$(OBJ)/e_%.o: engine/%.c $(ENG_HDRS) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
ENG_O     := $(MOVA_O) $(OBJ)/e_engine.o $(OBJ)/e_qwen4_api.o $(Q4E_O)
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
SEARCH_O  := $(OBJ)/search_metal.o $(OBJ)/search4_metal.o $(OBJ)/searchc_metal.o $(OBJ)/searchp_metal.o $(OBJ)/trellis_metal.o
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
$(BIN)/nslm-cpu-seed-bench: harness/nslm-cpu-seed-bench.c $(LIB_O) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -O3 $< $(LIB_O) -o $@ -lm -lpthread
$(BIN)/nslm-qwen4-pack: nslm/qwen4_pack.c $(LIB_O) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -O3 $< $(LIB_O) -o $@ -lm -lpthread
$(BIN)/nslm-qwen4-emulate: nslm/qwen4_emulate.c $(LIB_O) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -O3 $< $(LIB_O) -o $@ -lm -lpthread

# ---- tests (the same C sources as the CUDA build; test_mova_kernels runs on tests/kernel_backend_metal.m) ---------
$(BIN)/test_%: tests/test_%.c $(LIB_O) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(LIB_O) -o $@ -lm -lpthread
$(BIN)/test_chat_template $(BIN)/test_tool_calls $(BIN)/test_tokenizer $(BIN)/test_kv_disk $(BIN)/test_qwen4_template $(BIN)/test_qwen4_tools: $(BIN)/test_%: tests/test_%.c $(HARN_O) $(LIB_O) tests/data/template_golden.json
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
# the trellis encoder on Metal against nslm_tq_encode_row: the same test as the CUDA build's
$(BIN)/test_trellis_gpu: tests/test_trellis_gpu.c $(OBJ)/trellis_metal.o $(LIB_O) $(NSLM_HDRS) $(RES)/trellis.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(OBJ)/trellis_metal.o $(LIB_O) $(LIBS) -lpthread -lm -o $@
$(BIN)/test_search4_gpu: tests/test_search4_gpu.c $(OBJ)/search4_metal.o $(LIB_O) $(NSLM_HDRS) $(RES)/search4.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(OBJ)/search4_metal.o $(LIB_O) $(LIBS) -lpthread -o $@
$(BIN)/test_mova_kernels: tests/test_mova_kernels.c tests/kernel_backend_metal.m tests/kernel_backend.h engine/kernels_moe.metal \
                          engine/kernels_qwen4.metal $(LIB_O) $(RES)/kernels_moe.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c tests/test_mova_kernels.c -o $(OBJ)/test_mova_kernels.o
	$(CC) $(OBJCFLAGS) -Itests tests/kernel_backend_metal.m $(OBJ)/test_mova_kernels.o $(LIB_O) $(LIBS) -o $@
$(BIN)/test_qwen4_kernels: tests/test_qwen4_kernels.c tests/kernel_backend_metal.m tests/kernel_backend.h engine/kernels_moe.metal \
                           engine/kernels_qwen4.metal $(LIB_O) $(RES)/kernels_moe.metallib $(RES)/kernels_qwen4.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c tests/test_qwen4_kernels.c -o $(OBJ)/test_qwen4_kernels.o
	$(CC) $(OBJCFLAGS) -Itests tests/kernel_backend_metal.m $(OBJ)/test_qwen4_kernels.o $(LIB_O) $(LIBS) -o $@
$(BIN)/test_qwen4_metal: tests/test_qwen4_metal.c $(Q4E_O) $(LIB_O) $(NSLM_HDRS) $(RES)/kernels_moe.metallib $(RES)/kernels_qwen4.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(Q4E_O) $(LIB_O) $(LIBS) -lm -lpthread -o $@
# the MTP head on the Metal engine: the same test as the CUDA build's
$(BIN)/test_qwen4_mtp: tests/test_qwen4_mtp.c $(Q4E_O) $(LIB_O) $(NSLM_HDRS) $(RES)/kernels_moe.metallib $(RES)/kernels_qwen4.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(Q4E_O) $(LIB_O) $(LIBS) -lm -lpthread -o $@
# eng_generate with MTP drafts (Q4E_DRAFTS) through engine/engine.c: the same test as the CUDA build's
$(BIN)/test_qwen4_api_mtp: tests/test_qwen4_api_mtp.c $(ENG_O) $(LIB_O) $(ENG_HDRS) $(NSLM_HDRS) $(RES)/kernels_moe.metallib $(RES)/kernels_qwen4.metallib
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(ENG_O) $(LIB_O) $(LIBS) -lm -lpthread -o $@
$(BIN)/nslm-qwen4-gen: harness/nslm-qwen4-gen.c $(Q4E_O) $(HARN_O) $(LIB_O) $(HARN_HDRS) $(NSLM_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< $(Q4E_O) $(HARN_O) $(LIB_O) $(LIBS) $(HLIBS) -lm -lpthread -o $@

$(OUT)/test/affine/index.txt: tools/mova_affine_golden.py tools/mova_common.py
	$(PY) tools/mova_affine_golden.py --out $(OUT)/test/affine

# the MLX loaders (tools/nanoseedlm_k2.py; tools/nanoseedlm_trellis.py through nanoseedlm_qwen4.py, which needs cc); OMLX_K2_MODEL = oMLX's k2_horizon_model.py
test-mlx:
	$(PY) tools/test_nanoseedlm_k2.py
	$(PY) tools/test_nanoseedlm_trellis.py

# the MLX goldens of test_affine are built first (macOS has MLX); test_mova_cfg and test_tokenizer read MOVA_DIR (a model
# folder) and are skipped without one.  A test that exits 77 is skipped.
test: $(TESTS) $(OUT)/test/affine/index.txt
	@for t in $(TESTS); do echo "== $$t"; $$t; r=$$?; if [ $$r -eq 77 ]; then echo "(skipped)"; elif [ $$r -ne 0 ]; then exit 1; fi; done; echo "all tests passed"

clean:
	rm -rf $(OUT)
