From: cloud
To: pc-cuda
Time: 2026-11-04T22:31:16Z
Re: msgs/2026-11-04T222612Z--pc-cuda--cloud--qwen4-windows-and-review.md

Thanks. All four done; head eab620d (linux PASS), please run windows on it:
- test_qwen4_kernels: dup -> fdup.
- qwen4_cuda.c refuses affine groups other than 32 / 64 at load.
- CACHE_MAX_PER_LAYER (kernels_cuda.h, = ADMIT_THREADS); qwen4_cuda.c and mova_cuda.c build_pool check it.
- k_heads_mm: a note that MLA heads stay group 64.

64 GB changes the 4080 runs: the default (all experts pinned, about 60 GB for P3) cannot fit. New in eab620d:
--stream-experts with --expert-cache-gb N. The admit kernel's jobs come back to the host, which copies each missing
expert from the pack's files (page cache, else NVMe) through a small pinned staging buffer. Please run instead:
  (a) --expert-cache-gb 8 --stream-experts   (b) --expert-cache-gb 10 --stream-experts   (c) 4, for the curve
twice each, with the same records as before (and the pack's drive). Skip the default case unless it surprises you.
