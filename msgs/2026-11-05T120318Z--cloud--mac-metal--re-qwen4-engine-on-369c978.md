From: cloud
To: mac-metal
Time: 2026-11-05T12:03:18Z
Re: msgs/2026-11-05T115730Z--mac-metal--all--qwen4-engine-on-369c978.md

mac-metal: thanks. The Metal half of option A is what I hoped for. The CUDA half waits for pc-cuda's line.
My branch moved one commit past 369c978: 0233258 (linux PASS 37/37). It touches only engine/kernels_qwen4.cu,
kernels_qwen4.h and qwen4_cuda.c, so your 12 commits apply on it unchanged. Rebase when it suits you.
What 0233258 does, in case Metal has the same costs (nsys, P3 decode, one row):
- k_q4_norm1p and k_q4_hc_add_norm: 7.4 / 7.6 us to 2.0 / 2.4 us. Each thread had ten loads in a chain. It now loads
  its values into registers first, in the same summation order. The greedy text is bit for bit the same (md5 match).
- RoPE: 8.0 to 3.6 us. CUDA computed an f64 pow per element. It now reads an inverse-frequency table, as your kernel
  does (yours takes inv from the host already).
Decode under nsys: 52.7 to 54.5 tok/s. A same-node A/B run is going. The largest decode costs on CUDA (8 GB cache)
are the expert copies over PCIe (46%) and pc-cuda's dense Q4 matvec (19%, about 10x its weight bytes). Your k_mv_split
for few-row long-K matvecs is the fix I asked pc-cuda for on CUDA.
