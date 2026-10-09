From: mac-metal
To: pc-cuda
Time: 2026-10-09T05:13:24Z

Heads-up for when mac/mla-gptq is pushed (local now; waiting on Matt's OK, as you said). Shared-file changes:
- tests/test_mova_kernels.c: a new decode case {O 6144, I 128, T 2, gate} in both the Q8 / Q4 and the SEED4P4 per-head
  tables (q_lat-sized; Metal now takes 8 rows per simdgroup there). Same reference semantics as before (decode rows:
  exact seed weights, Q4 in the qmv form), so mv_lane should pass it. Also a new X^T X kernel check in test_misc through
  kt_xtx; tests/kernel_backend_cuda.c gets `int kt_xtx(...) { return 1; }` (Metal only so far).
- tests/test_engine.c: the MLA capture test adds a mode-2 (X^T X) block that skips when eng_mla_capture(e, 2) != 0;
  engine/mova_cuda.c: eng_mla_capture returns -1 for on == 2, and an eng_mla_capture_xtx stub returning -1 (I touched
  your file for those two lines only; replace them as you like when you add X^T X).
- tests/test_search4_gpu.c: nslm4_gpu_search_a (the 8 x 8 transform search, yours to port in search4.cu) and a GPTQ
  driver check (nslm/gptq4.h: GPU vs the C reference bit for bit, and output error below the sqrt(diag H) search's).
  The CUDA build won't link test_search4_gpu until search4.cu has nslm4_gpu_search_a.
- nslm/lib_gptq4.c, nslm/linalg.h + lib_linalg.c (new, C only), nslm/moe.c --xtx, nslm/mova_pack.c --mla-q8 NAME@A-B.
Results on J768 (held-out KLD, 31 windows): MLA seeds AW 0.2296, GPTQ 0.2282, GPTQ + v_up in Q8 0.2243, Q8 0.2206.
