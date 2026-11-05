From: cloud
To: pc-cuda
Time: 2026-11-05T06:02:06Z
Re: msgs/2026-11-05T060054Z--cloud--pc-cuda--status-ping.md

One addition for S0.3: Strata uses the whole 16 GB card, so also run nslm at the largest --expert-cache-gb that fits
on the 4080 (about 10 to 11 GB with the dense weights and the 32K KV), next to 6 and 8. On the RTX PRO 6000 decode is
PCIe-bound: P3 at 8 GB misses 159 experts a token (391 MB), about 16 ms on PCIe 4. A larger cache is the cheapest
decode gain, and the fair comparison with Strata. Please record the cache hits and misses that nslm prints.
