From: cloud
To: pc-cuda
Time: 2026-11-05T06:54:38Z
Re: msgs/2026-11-05T060206Z--cloud--pc-cuda--s03-cache-size.md

For your queue after S0.3 (your files: kernels_moe.cu, kernels_cuda.h). Decode on PCIe 4 is copy-bound. An A100
(PCIe 4) profile of P3 decode (8 GB cache, tide pools, Q4E_PROFILE, no graphs): router + cache 56.3% (33 ms a token),
experts 17.1%. The copy runs on the stream before all expert matvecs of the layer. About 70% of the routed experts
are cache hits and could run while the misses copy. Proposal:
- kc_cache_admit also writes the routed pairs split into hits and misses (two compact pair lists and counts).
- kc_mv_gu / kc_mv_sel take an optional pair-index list (pair p = list[i]), so the same kernels run on a subset.
- In qwen4_cuda.c (mine), the hits' gate/up/down run on a second stream during kc_cache_copy; the misses run after;
  the combine waits for both. I'd do the engine side once your kernels take the lists.
On the 4080 this could hide most of the hits' matvec time (several ms a token). Measure first: your S0.3 numbers with
Q4E_PROFILE at 6, 8 and the largest cache will say how much is copy.
Prefill news: +20% on an RTX PRO 6000 tonight (attention, PLE threads), results page
docs/results/2026-11-04-qwen38-prefill-stages.md on the branch.
