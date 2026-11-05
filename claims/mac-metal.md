# mac-metal claims (updated 2026-11-05T14:05:41Z)

- **Now (2026-11-05):** the Metal qwen4 engine, `mac/qwen4-engine` 9439cce (on claude/sleepy-brown-k67z9u 483a91a):
  macos PASS, linux PASS; windows asked of pc-cuda. It lands after cloud's 483a91a, then needs Matt's go for main.
  Files: engine/qwen4_gpu.m, engine/kernels_qwen4.metal, engine/kernels_moe.metal, engine/mova_gpu.m (the option A
  include line), tests/kernel_backend_metal.m (mine); tests/test_qwen4_metal.c, Makefile, docs/log.md (shared).
  P3 on the M4 Max: decode 42-43 tok/s short, 36-38 after 19.8K tokens; prompt 400-440 tok/s (mlx-vlm 0.7.12: 40.1,
  457-500). nslm-chat and nslm-serve run it.
- Next (mine): QSA indexer at long context; prompt GEMM tile loads (seed decode / dequant, occupancy); k_mv's multi-token
  path equal to its one-token path (before MTP on Metal). The log's commit IDs get fixed once at merge time.
- Later: llama.cpp (Metal) on UD-Q4_K_XL as the Mac baseline (111 GB download); nslm/searchc_metal.m is mine on main (stub).
