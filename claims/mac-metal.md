# mac-metal claims (updated 2026-11-05T11:57:32Z)

- **Now (2026-11-05):** the Metal qwen4 engine, `mac/qwen4-engine` 89093ce (on claude/sleepy-brown-k67z9u 369c978):
  macos PASS, linux PASS; windows asked of pc-cuda. It lands after cloud's branch, then needs Matt's go for main.
  Files: engine/qwen4_gpu.m, engine/kernels_qwen4.metal, engine/kernels_moe.metal, engine/mova_gpu.m (the option A include
  line), tests/kernel_backend_metal.m (mine); tests/test_qwen4_metal.c, Makefile, docs/concepts/parity.md, docs/log.md (shared).
  P3 on the M4 Max: decode 42-43 tok/s, 5.6K-token prompt 414-439 tok/s (mlx-vlm 0.7.12: 40.1, 457-500).
- Next (mine): prompt GEMM tile loads (seed decode / dequant, occupancy); seed gather speed; k_mv's multi-token path
  equal to its one-token path (before MTP on Metal).
- Later: llama.cpp (Metal) on UD-Q4_K_XL as the Mac baseline (111 GB download); nslm/searchc_metal.m is mine on main (stub).
