# cloud claims (updated 2026-11-05T11:08:12Z)

# cloud claims (updated 2026-11-05T11:20:00Z)

- **cloud** is a Claude Code cloud session (no GPU; HF Jobs for Linux and GPU runs). Matt's go: Qwen3.8-Flash-Next.
- **Branch:** `claude/sleepy-brown-k67z9u`. Fast-forward target for main: 812c516 (linux and macos PASS, windows open).
- **Done tonight:** the qwen4 CUDA prefill and decode speed work (docs/results/2026-11-05-qwen38-*.md).
  - Mine: `engine/qwen4_cuda.c`, `engine/qwen4_eng.h`, `engine/kernels_qwen4.*`, `harness/nslm-qwen4-*.c`,
    `nslm/*qwen4*`, `tests/test_qwen4_*.c`, `tests/data/qwen4_tiny_attn`, `tools/*qwen4*`, the qwen4 results pages.
- **Now (Matt: option A):** nslm-chat / nslm-serve for Qwen3.8-Flash-Next, one binary. Step 1, new files of mine:
  `engine/qwen4_api.c` (engine_api.h over the qwen4 engine, one slot), `engine/engine_qwen4.h` (its eng_* names as
  q4_eng_*), `tests/test_qwen4_api.c`. Step 2 (shared, later, I will ask first): `engine/engine.c` picks the engine by
  model_type; the MoVA engine files take the names mova_eng_* through one include (their owners, or the build files).
