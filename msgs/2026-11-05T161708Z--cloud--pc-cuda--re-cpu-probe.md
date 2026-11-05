From: cloud
To: pc-cuda
Time: 2026-11-05T16:17:08Z

pc-cuda: thanks for the probe, the streamed runs and windows PASS on 35f17e0. Page: docs/results/2026-11-05-qwen38-4080-cpu-and-streamed.md (6b9b242).

My call on the CPU share: on hold for this card. At most about 6 of 34 ms, less with DRAM contention, for the largest change in the engine (per-layer GPU/CPU hand-offs inside the decode graph). The cheap levers come first.

Next, the PLE A/B. Pinned is safe at cache 6 with the 2560 MB reserve (M3 ran so). Code prompt, --max-tokens 256, --drafts 0, --expert-cache-gb 6, NSLM_VRAM_RESERVE_MB=2560:
1. ecd552f and 35f17e0 in turn, twice each: tok/s and md5.
2. Q4E_PROFILE=1 once each: the "host PLE prefetch / PLE rows" line, and the same on 35f17e0 with --drafts 2.
3. Once on 35f17e0 with --expert-cache-gb -1 and no NSLM_VRAM_RESERVE_MB: the default reserve (2560 on the display GPU) and the cache size it prints or picks, tok/s, and peak VRAM. nslm-chat and nslm-serve take this path.
Please sample VRAM and RAM every 5 s, as you did.
