# mac-metal claims (updated 2026-11-05T02:35:15Z)

- **Done (2026-11-05T02:33Z):** the Metal qwen4 kernels are on main (fast-forward to mac/qwen4-metal c404931, Matt's go,
  green on macos, linux and windows).
- **Now:** macos runs for the others: cloud's rebased claude/sleepy-brown-k67z9u head (after 4e73cb7), and pc-cuda's
  pc/fp4-decode-gap when it is pushed. No shared files are claimed.
- **Next (mac-metal's):** a Metal qwen4 engine (nslm-qwen4-gen on Metal), with the MoE combine, the stream tile and the
  Q4 embedding on Metal, plus QSA top-k in block order (cloud's qsa-order note). I will claim its files when it starts.
- Later: llama.cpp (Metal) on UD-Q4_K_XL as the plan's Mac baseline; the mova_gpu.m rename #include (option A).
