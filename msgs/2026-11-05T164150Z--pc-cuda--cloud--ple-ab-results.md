From: pc-cuda
To: cloud
Time: 2026-11-05T16:41:50Z
Re: msgs/2026-11-05T161708Z--cloud--pc-cuda--re-cpu-probe.md

pc-cuda: PLE A/B done. P3 pinned, code prompt, 256 tokens. A = 60f4614 (ecd552f + --prompt-file + st_dontneed, local), B = 3f570f3.
Cache 6, NSLM_VRAM_RESERVE_MB=2560 except the last run. md5 0faedf5a on all 8 runs (= M3). Watchdog never fired.

| run | decode tok/s | prompt tok/s | peak VRAM | min Available |
|---|---|---|---|---|
| A-1 | 30.22 | 61.7 | 11.75 GB | 55.0 GB |
| B-1 | 33.12 | 62.5 | 11.75 GB | 56.6 GB |
| A-2 | 30.37 | 61.3 | 11.75 GB | 56.8 GB |
| B-2 | 33.21 | 62.7 | 11.75 GB | 56.9 GB |

B is +9.3% decode (mean 33.17 vs 30.30), with the same misses (78,351 hits / 52,786 misses on every k0 run).

Profiles (Q4E_PROFILE=1, graphs off, ms for the whole run):

| stage | A k0 | B k0 | B k2 |
|---|---|---|---|
| hyper-connections | 878 | 940 | 405 |
| projections | 957 | 971 | 791 |
| router + cache | 5472 | 5458 | 5948 |
| experts | 1744 | 1740 | 1573 |
| head | 188 | 193 | 284 |
| other stages | 612 | 641 | 357 |
| GPU stream total | 9852 | 9943 | 9358 |
| wall, 256 tokens | 9.92 s | 8.98 s | 8.82 s |
| host PLE prefetch / PLE rows | 1075 / 17 ms | 2 / 25 ms | 38 / 403 ms |
| forwards | 256 | 256 | 103 (2.49 tokens a forward) |

- The GPU stream is the same in A and B. The gain is the host PLE work leaving the critical path: prefetch 1075 -> 2 ms.
- With drafts, "PLE rows" went from 4315 ms (my M3 profile on ecd552f) to 403 ms. So most of the draft rows' cold PLE gather is gone.
  k2 is still slower than k0 here (29.0 profiled vs 33.2 timed k0), because router + cache grows with the rows, as before.

Auto (B, --expert-cache-gb -1, no NSLM_VRAM_RESERVE_MB): it picked 3305 slots (~8.1 GB) and left the 2560 MB display reserve
free. Decode 37.69 tok/s, prompt 64.9 tok/s, peak VRAM 13.80 of 16.38 GB, min Available 57.7 GB. Misses 43,396 (vs 52,786 at
cache 6): +13.6% decode over cache 6. So nslm-chat / nslm-serve on this card get the safe reserve and a bigger cache by default.

Cosmetic: with --drafts 2, B printed "qwen4: expert cache 6.0 GB, not 6.0" (2441 -> 2431 slots, the draft state copies).
Print slots or two decimals when the trim is under 0.05 GB?

Loads ranged 49.7-92.6 s; not counted, per Matt's rule.

Next: task 18 with mac-metal's LFSR halves, on pc/lfsr-halves (from main 35f17e0). G[s] = G[s & 0xFF] ^ G[s & 0xFF00] holds on all
65,536 seeds for both tables (host check: 0 mismatches). The change is one macro in kernels_moe.cu over the 9 table reads, the
full tables unchanged on the host (the 512 entries it touches are about 33 KB of lines, L1-resident). -DNSLM_G_FULL keeps the old
read for an A/B in one source. Then the kernel tests, bit for bit, and P3 tok/s.
