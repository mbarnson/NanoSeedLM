# mac-metal claims (updated 2026-11-05T05:46:49Z)

- **Now (2026-11-05):** the Metal qwen4 engine, branch `mac/qwen4-engine` from main 047ad9f
  (worktree ~/devel/nslm-mac-q4e). It implements engine/qwen4_eng.h on Metal, unchanged (q4e_open / forward / reset
  / close), with the kernels on main (kernels_qwen4.metal, kernels_moe.metal: affine Q4 g32 and the P3 / P4 seed
  matvec and GEMM). Unified memory: the routed experts are read in place from the mapped folder, with no expert cache;
  cache_gb / stream_experts are ignored on the Mac.
  Files: engine/qwen4_gpu.m (new, mine), engine/*.metal (mine, if a kernel is missing); tests/test_qwen4_metal.c
  (new: the engine against tests/data/qwen4_tiny_golden.safetensors with test_qwen4_engine.c's limits; that file
  stays CUDA-only and unchanged); Makefile (shared: builds nslm-qwen4-gen and the test on macOS);
  docs/concepts/parity.md, docs/log.md (shared). harness/nslm-qwen4-gen.c and engine/qwen4_eng.h: no change.
  Then P3 / P4 decode and prompt speed on the M4 Max against mlx-vlm, and the QSA top-k in block order.
- Done: the MTP study (msgs/2026-11-05T054253Z--mac-metal--cloud--mtp-study-results.md).
- Later: llama.cpp (Metal) on UD-Q4_K_XL as the Mac baseline; the mova_gpu.m rename #include (option A);
  nslm/searchc_metal.m is mine once on main (stub today).
