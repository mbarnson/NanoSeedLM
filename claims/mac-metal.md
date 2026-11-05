# mac-metal claims (updated 2026-11-05T05:22:48Z)

- **Now (2026-11-05, cloud's next-perf-work):** the MTP study on the M4 Max. mlx-vlm 0.7.12, greedy, 256 tokens;
  P3 pack and oQ4e-mtp (4-bit MLX) as targets, each with its own MTP head; no drafts / 1 / 2 / 3 / adaptive; a chat and
  a code prompt; two runs each. Local only, no shared files. Results in a message.
- macos runs for the others as asked.
- **Next (mac-metal's):** a Metal qwen4 engine (nslm-qwen4-gen on Metal), with the MoE combine, the stream tile and the
  Q4 embedding on Metal, plus QSA top-k in block order. I will claim its files when it starts.
- Later: llama.cpp (Metal) on UD-Q4_K_XL as the Mac baseline; the mova_gpu.m rename #include (option A);
  nslm/searchc_metal.m is mine once on main (stub today).
