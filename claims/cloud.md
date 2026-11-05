# cloud claims (updated 2026-11-05T12:44:30Z)

# cloud claims (updated 2026-11-05T12:45:00Z)

- **cloud** is a Claude Code cloud session (no GPU; HF Jobs for Linux and GPU runs). Matt's go: Qwen3.8-Flash-Next.
- **Branch:** `claude/sleepy-brown-k67z9u` ba2068e (linux running). Fast-forward target for main: ba2068e once
  windows and macos pass it.
- Mine: `engine/qwen4_cuda.c`, `engine/qwen4_eng.h`, `engine/kernels_qwen4.*`, `engine/qwen4_api.c`,
  `engine/engine.c`, `engine/engine_qwen4.h`, `engine/engine_mova.h`, `engine/engine_unname.h`,
  `harness/nslm-qwen4-*.c`, `harness/chat_qwen.c`, `harness/chat_model.*`, `nslm/*qwen4*`, `tests/test_qwen4_*.c`,
  `tests/test_engine_dispatch.c`, `tests/data/qwen4_tiny_attn`, `tools/*qwen4*`, the qwen4 results pages.
  For the Qwen paths: `harness/nslm-chat.c`, `harness/nslm-serve.c`, `harness/tool_calls.c` (tell me if you need them).
- **Done today:** decode 54.6 to 58.3 tok/s (short), 43.8 to 58.8 (after 16K tokens), the same decode KL.
  nslm-chat / nslm-serve run Qwen3.8-Flash-Next end to end in a preview with the option A wiring.
- **Waiting:** pc-cuda's include line in mova_cuda.c (then engine.c joins CMake, and the guide). Matt's go on the MTP
  and adaptive-tier plans.
