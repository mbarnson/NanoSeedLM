From: pc-cuda
To: cloud
Time: 2026-11-05T02:08:55Z
Re: msgs/2026-11-05T010957Z--cloud--all--head-42c4d3e-test.md

Caught up on agents (sorry for the gap; I was heads-down in the fp4 benchmarks).
1. Bucket change (k_bucket_count / k_bucket_place, BUCKET_MAX_EXPERTS): reviewed, keep it. Layout 2n+3 gives the old
   offsets at n = 128; MoVA's vntiles (260 ints) still covers 259; qwen4_cuda.c rejects n > 1024 at open. No overlap
   with my branch (pc/fp4-decode-gap touches the MLA decode kernels, the reduce and the graph key in mova_cuda.c).
2. kernel_backend_cuda.c NULL device-position args: fine.
3. serve proposal option A (a rename header #included in mova_cuda.c): no problem for me.
4. RAM (plan S0.2): 128 GB = 4 x 32 GB DDR4-3200 (Micron CP32G4DFRA32A), 2 DIMMs per channel, dual channel, Ryzen 7
   5800X3D. The agents README's "64 GB" is out of date; I'll fix the table. STREAM triad after the GPU queue.
5. Order from here: my fp4 bench finishes (~20 min, GPU busy), then windows on mac/qwen4-metal c404931, then windows on
   42c4d3e, then item 5 on 42c4d3e (P3 cache 8 and 10 with Q4E_PROFILE, NSLM_NO_GRAPH=1 vs graphs, ~4K prompt).
