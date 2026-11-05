# cloud claims (updated 2026-11-05T11:35:32Z)

# cloud claims (updated 2026-11-05T11:40:00Z)

- **cloud** is a Claude Code cloud session (no GPU; HF Jobs for Linux and GPU runs). Matt's go: Qwen3.8-Flash-Next.
- **Branch:** `claude/sleepy-brown-k67z9u` d0ccf59 (linux PASS). Fast-forward target for main: d0ccf59 once
  windows and macos pass it (812c516 is still open on windows).
- Mine: `engine/qwen4_cuda.c`, `engine/qwen4_eng.h`, `engine/kernels_qwen4.*`, `engine/qwen4_api.c`,
  `engine/engine.c`, `engine/engine_qwen4.h`, `engine/engine_mova.h`, `engine/engine_unname.h`,
  `harness/nslm-qwen4-*.c`, `harness/chat_qwen.c`, `nslm/*qwen4*`, `tests/test_qwen4_*.c`,
  `tests/test_engine_dispatch.c`, `tests/data/qwen4_tiny_attn`, `tools/*qwen4*`, the qwen4 results pages.
- **Now (Matt: option A):** nslm-chat / nslm-serve for Qwen3.8-Flash-Next. Step 2 waits for the include line in
  mova_cuda.c (pc-cuda) and mova_gpu.m (mac-metal). Step 3, now: the Qwen paths in `harness/nslm-chat.c`,
  `harness/nslm-serve.c` and `harness/tool_calls.c` (nobody claims them; tell me if you have work there), and a new
  `harness/chat_model.c` (template, stop tokens, thinking tags and tool calls by model).
