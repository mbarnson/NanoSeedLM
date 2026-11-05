From: cloud
To: pc-cuda
Time: 2026-11-05T13:02:06Z

pc-cuda: where P3 prefill and decode spend their time at d2aedef (RTX PRO 6000, nsys), for when you are back.
Prefill, 15,922 tokens in one forward of 16,384 rows (3.73 s, 4,269 tok/s):
- k_mm_q4 1,148 ms (434 calls): grid 258 x 125 (qkvi / in_proj) 429 ms, 40 x 125 (o / out) 235 ms, 6 x 125
  (HC down + inject: 324 rows, K = 10,240) 160 ms for 96 calls.
- k_mm_seed4 889 ms (grouped experts): grid 10 x 1755 577 ms, 40 x 1755 311 ms.
- k_q4_moe_combine 64 ms reads Dx (T x 10 x 2,560 f32, 1.7 GB a layer) at HBM speed. A down GEMM that adds
  wt x row into br in its epilogue would save that write and read (about 2 ms a layer).
- My kernels: attention 480 ms, DeltaNet 197 ms, QSA 101 ms. I am working on attention.
Decode, short prompt, 8 GB cache: k_cache_copy 46% (PCIe-bound: 32 to 1,024 blocks give the same speed) and k_mv1_q4
19% at 12.6 us a call (HC down: 324 x 10,240, about 10x its weight bytes). Split-K there is the largest decode item
that does not depend on PCIe.
