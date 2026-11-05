# mac-metal claims (updated 2026-11-05T07:14:18Z)

- **Now (2026-11-05):** the Metal qwen4 engine on `mac/qwen4-engine` 48718c6abb15 (from main 047ad9f): macos PASS,
  linux PASS; windows asked of pc-cuda; then Matt's go to fast-forward main. Files: engine/qwen4_gpu.m (new),
  engine/kernels_qwen4.metal, engine/kernels_moe.metal (mine); tests/test_qwen4_metal.c (new), Makefile,
  docs/concepts/parity.md, docs/log.md (shared). P3 decode on the M4 Max: 38.4 tok/s warm (mlx-vlm: 40.1).
- Next (mine): decode profile against mlx-vlm; the prompt path; k_mv's multi-token path equal to its one-token path
  (needed before MTP on Metal).
- Later: llama.cpp (Metal) on UD-Q4_K_XL as the Mac baseline; the mova_gpu.m rename #include (option A);
  nslm/searchc_metal.m is mine once on main (stub today).
