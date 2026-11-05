# cloud claims (updated 2026-11-05T02:21:48Z)

# cloud claims (updated 2026-11-05T02:09:19Z)

# cloud claims (updated 2026-11-05T02:10:00Z)

- **cloud** is a Claude Code cloud session (no GPU; HF Jobs for Linux and GPU runs). Matt's go: Qwen3.8-Flash-Next.
- **Now:** the Qwen3.8-Flash-Next CUDA engine speed, on `claude/sleepy-brown-k67z9u` (head 25ae5f932f8c, from main
  878d4d4). Matt's go: open the PR when the head is green.
  - Mine: `engine/qwen4_cuda.c`, `engine/qwen4_eng.h`, `engine/kernels_qwen4.*`, `harness/nslm-qwen4-*.c`,
    `nslm/*qwen4*`, `tests/test_qwen4_*.c`, `tools/hf_qwen4_*.sh`, `tools/qwen4_cache_sim.py`, the qwen4 results pages.
  - Shared files this branch changes: `README.md`, `llms.txt`, `nslm/format.h`, `nslm/rows.h`, `nslm/ple.h` and their
    lib files (new functions only), `tests/test_ple.c`, `harness/nslm-mova-kbench.c` (qwen4 cases at the end),
    `docs/reference/env.md`, `docs/reference/tools.md`, `docs/plans/2026-11-04-qwen38-flash-next.md` (S8 steps),
    `docs/log.md`, `docs/index.md`.
  - pc-cuda's files, small: `engine/kernels_moe.cu`, `engine/kernels_cuda.h` (the bucket kernels only),
    `tests/kernel_backend_cuda.c` (the qwen4 section only: a NULL device-position argument in three calls).
- **Next, after the speed work (Matt: option A):** nslm-chat / nslm-serve for Qwen3.8-Flash-Next, one binary.
