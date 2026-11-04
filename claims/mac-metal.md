# mac-metal claims (updated 2026-11-04T23:31:00Z)

- **Now (2026-11-04, Matt's go): the Metal qwen4 kernels**, at parity with engine/kernels_qwen4.cu, on `mac/qwen4-metal`
  (from main db0ea15). Each kernel is checked against nslm/qwen4_ops.h by tests/test_qwen4_kernels.c through
  tests/kernel_backend_metal.m, until it prints no "not on Metal":
  1. Affine Q4 group 32 in the Metal matvec, GEMM and embedding (kernels_moe.metal).
  2. GDN (conv, step, norm), norm1p, hyper-connections (act, mix, gate, add, tile), head norm + RoPE, the router, the
     MoE combine, PLE gate + conv, QSA (block mean, scores, top-k), attention: a new engine/kernels_qwen4.metal.
- Files: engine/kernels_qwen4.metal (new), engine/kernels_moe.metal, tests/kernel_backend_metal.m (mine); Makefile,
  docs/concepts/parity.md (shared). The arg structs in kernels_moe.metal are CUDA's too: I keep their layout, or I
  message pc-cuda first.
- Next, as its own claim: a Metal qwen4 engine (nslm-qwen4-gen on Metal), checked against mlx-vlm and the C reference.
