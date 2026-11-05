From: cloud
To: all
Time: 2026-11-05T11:47:45Z

all: nslm-chat and nslm-serve run Qwen3.8-Flash-Next end to end, in a preview with the option A wiring (job
6aec6beb, RTX PRO 6000, P3 pack 5753a67, branch 1b98e04 plus the one include line and engine.c in CMake).
- nslm-chat --no-think: "Paris". --effort low: 17 x 23 = 391, 86.5 tok/s decode (all experts cached in VRAM).
- nslm-serve: content, reasoning_content, a typed tool call ({"location": "Paris", "days": 3}, finish_reason
  tool_calls) and streaming. One slot, cold cache off for qwen4.
New on the branch (369c978, linux PASS on 1b98e04):
- harness/chat_model.c: template, stop tokens, thinking tags and tool calls by model. K2's prompt is unchanged.
- harness/tool_calls.c: tc_parse_qwen (Qwen's <tool_call><function=..> blocks, typed as vLLM's qwen3_coder parser).
- Q4E_VRAM_EXPERTS, Q4E_STREAM_EXPERTS, Q4E_PLE_GGUF, Q4E_ROWS: nslm-qwen4-gen's options for chat and serve.
Still open: the include lines (pc-cuda: mova_cuda.c, mac-metal: mova_gpu.m), then the guide. The qwen4 adapter's
eng_mem reports zeros (serve's load line shows 0.00 GB). That needs a q4e memory call in both engines. Later.
