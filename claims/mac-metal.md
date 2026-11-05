# mac-metal claims (updated 2026-11-05T02:31:19Z)

- **Now: the Metal qwen4 kernels** on `mac/qwen4-metal` c404931f1ac4 (rebased on main 878d4d4). macos PASS, linux PASS;
  windows queued on pc-cuda. Matt's go: fast-forward main to it when windows passes (mac-metal does it and announces).
  Files: engine/kernels_qwen4.metal (new), engine/kernels_moe.metal, tests/kernel_backend_metal.m (mine); Makefile,
  docs/concepts/parity.md, docs/log.md (shared).
- Next (not yet claimed): a Metal qwen4 engine (nslm-qwen4-gen on Metal), with the MoE combine, the stream tile and the
  Q4 embedding on Metal. It includes QSA top-k in block order on Metal (cloud's qsa-order note).
- Later (Matt: option A, after the speed work): the one-line rename #include in engine/mova_gpu.m, when cloud sends it.
