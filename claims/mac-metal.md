# mac-metal claims (updated 2026-11-04T23:43:48Z)

- **Now: the Metal qwen4 kernels** on `mac/qwen4-metal` 611254c (from main db0ea15). macos PASS, linux PASS; windows
  asked of pc-cuda; then Matt's go to fast-forward main. Files: engine/kernels_qwen4.metal (new), engine/kernels_moe.metal,
  tests/kernel_backend_metal.m (mine); Makefile, docs/concepts/parity.md, docs/log.md (shared).
- Next (not yet claimed): a Metal qwen4 engine (nslm-qwen4-gen on Metal), with the MoE combine, the stream tile and the
  Q4 embedding on Metal.
