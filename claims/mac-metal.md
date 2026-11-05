# mac-metal claims (updated 2026-11-05T13:35:55Z)

- **Now (2026-11-05):** the Metal qwen4 engine, `mac/qwen4-engine` 3fefabc (on claude/sleepy-brown-k67z9u 6198b48):
  macos PASS, linux PASS; windows asked of pc-cuda. It lands after cloud's 6198b48, then needs Matt's go for main.
  Files: engine/qwen4_gpu.m, engine/kernels_qwen4.metal, engine/kernels_moe.metal, engine/mova_gpu.m (the option A
  include line), tests/kernel_backend_metal.m (mine); tests/test_qwen4_metal.c, Makefile, docs/log.md (shared).
  P3 on the M4 Max: decode 42-43 tok/s short, 36-38 after 19.8K tokens; prompt 400-440 tok/s (mlx-vlm 0.7.12: 40.1, 457-500).
- Next (mine): QSA indexer at long context; prompt GEMM tile loads (seed decode / dequant, occupancy); k_mv's multi-token
  path equal to its one-token path (before MTP on Metal).
- Later: llama.cpp (Metal) on UD-Q4_K_XL as the Mac baseline (111 GB download); nslm/searchc_metal.m is mine on main (stub).
