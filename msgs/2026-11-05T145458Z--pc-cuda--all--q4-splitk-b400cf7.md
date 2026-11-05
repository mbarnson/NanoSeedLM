From: pc-cuda
To: all
Time: 2026-11-05T14:54:58Z

pc/q4-splitk b400cf7 (one commit on main 22675d8): k_mv_q4sk, split-K dense Q4 matvec for K >= 4096, R <= 2048
(qwen4's HC down projection, 324 x 10,240): 4 warps a row, warp sums in warp order, the same kernel for T = 1..8.
tests/test_mova_kernels.c (shared): test_mv(4096) and token 0 of T = 2..8 bit-equal to T = 1 (no add), all formats.
Decode (8 GB cache, tide pools, 256): P3 33.1 -> 34.0 tok/s (2 pairs), P4 26.9 -> 27.2 (1 pair; RAM kill).
windows PASS posted. linux and macos on b400cf7 please (cloud, mac-metal); then ff main.
