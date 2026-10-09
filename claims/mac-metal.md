# mac-metal claims (updated 2026-10-09T01:42:42Z)

- **Now, on `mac/mla-seeds` (from mla-memory f612209), Matt's go:** P=4 seeds for the MLA projections.
  1. Capture: `engine_api.h` gains eng_mla_capture / eng_mla_capture_read (per layer, per-column sums of squares of
     the MLA projections' inputs: attention input, value experts' v, heads' queries, latent outputs); Metal implements
     it; the CUDA engine returns -1 until you want it (I'll add only that stub to mova_cuda.c). New tool
     `harness/nslm-mova-mlacapture.c` (Makefile / CMake) over P-01's calibration text.
  2. `nslm/moe.c`: `--scope mla` (blocks per (layer, MLA tensor); per-head maps searched head by head).
  3. `nslm/mova_pack.c`: `--mla p4` (SEED4P4 from those blocks).
  4. Metal: SEED4P4 per-head maps (decode matvec, per-head GEMM, transposed q_lat). LINEAR MLA tensors already run
     SEED4P4 on both engines. CUDA per-head SEED4P4 would be yours, after pc/mla-heads-q.
