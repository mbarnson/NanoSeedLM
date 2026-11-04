# cloud claims (updated 2026-11-04T22:24:21Z)

- **cloud** is a Claude Code cloud session (no GPU; HF Jobs for Linux and GPU runs). Matt's go: Qwen3.8-Flash-Next.
- Branch `claude/sleepy-brown-k67z9u`, PR #15 (https://github.com/mbarnson/NanoSeedLM/pull/15): the qwen4 seed search,
  packs and MLX loader, a C reference, and a CUDA engine (`engine/qwen4_eng.h`, `nslm-qwen4-gen`, `nslm-qwen4-kld`).
- Files of other owners this branch changes (small; please review): `engine/kernels_moe.cu`, `engine/kernels_cuda.h`,
  `engine/mova_cuda.c`, `tests/kernel_backend_cuda.c` (pc-cuda); `engine/mova_gpu.m`, `tests/kernel_backend_metal.m`
  (mac-metal). New files only otherwise, plus shared docs, CMakeLists.txt and the Makefile.
- Not claimed: the Metal qwen4 kernels (open for mac-metal).
