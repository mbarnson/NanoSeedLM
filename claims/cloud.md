# cloud claims (updated 2026-11-05T00:11:14Z)

# cloud claims (updated 2026-11-05T00:15:00Z)

- **cloud** is a Claude Code cloud session (no GPU; HF Jobs for Linux and GPU runs). Matt's go: Qwen3.8-Flash-Next.
- **Now:** the Qwen3.8-Flash-Next CUDA engine speed, on `claude/sleepy-brown-k67z9u` (from main 878d4d4). Matt's go: open
  the PR when the head is green.
  - Mine: `engine/qwen4_cuda.c`, `engine/qwen4_eng.h`, `engine/kernels_qwen4.*`, `harness/nslm-qwen4-*.c`,
    `nslm/*qwen4*`, `tests/test_qwen4_*.c`, `tools/hf_qwen4_*.sh`, the qwen4 results pages.
  - Shared files this branch changes: `README.md`, `llms.txt` (the supported models), `nslm/format.h`, `nslm/rows.h`,
    `nslm/ple.h` and their lib files (row prefetch helpers, new functions only), `tests/test_ple.c`,
    `docs/reference/env.md`, `docs/log.md`, `docs/index.md`.
  - pc-cuda's files, small: `engine/kernels_moe.cu` and `engine/kernels_cuda.h`, the bucket kernels only (up to 1,024
    experts). pc-cuda: review it, or tell me to drop it.
- **Next:** qwen4 decode speed on the 4080 (plan S8): CUDA graphs for decode, then MTP drafts.
