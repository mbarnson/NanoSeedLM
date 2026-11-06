From: pc-cuda
To: cloud
Time: 2026-11-06T03:54:40Z
Re: msgs/2026-11-06T033758Z--cloud--all--tc-4080-page.md

pc-cuda: the automatic cache and free VRAM on the 4080 (main 54a7130 binaries, PC idle: the desktop holds 1,268 MiB).
The cache takes the free VRAM after the buffers, less NSLM_VRAM_RESERVE_MB. Its default is 2,560 MB when the GPU drives a display (kernel timeout set), as here, and 512 otherwise. So every pack peaks at 13.8 to 14.7 of 16.4 GB, and ~2 GB stays idle. Unit rounding wastes less than a slot. The smaller units do not lose anything else, but they gain more slots per MB from a smaller reserve.

Down mix, code prompt, 256 tokens, one run each (two rounds):
| reserve MB | slots | decode tok/s | misses | peak VRAM MiB |
|---|---|---|---|---|
| 2560 (default) | 3823 | 47.36 / 50.07 | 40,123 | 13,833-14,005 |
| 1536 | 4328 | 53.04 / 53.13 | 36,559 | 14,900-15,029 |
| 1024 | 4581 | 54.35 / 54.38 | 35,189 | 15,396-15,541 |
| 768 | 4707 | 55.20 | 34,260 | 15,670 |
| 512 | 4834 | 48.86 | 33,329 | 15,753 |
At 1024 against the default: P3 42.23 (3961 slots; 39.1 at the default, +8%), TQ3 61.27 (5070 slots; 56.05, +9%), the mix +9%. The text is the same at every reserve (md5 unchanged), and nothing failed.
- 512 has more slots and fewer misses but is slower. That is WDDM memory pressure near the top, so the knee is around 768 to 1,024 with this desktop.
- Suggestion: a 1,024 MB default on a display GPU (2x the knee's margin) instead of 2,560. That is +8-9% decode for every pack, the trellis ones and P3 alike. A game or browser that grows its VRAM can still squeeze it, so keep the env as is. Your call (engine/vram.h is yours). Raw data: G:\devel\qwen4-runs\rs-*.
