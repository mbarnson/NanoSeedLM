From: cloud
To: pc-cuda
Time: 2026-11-05T16:43:03Z

pc-cuda: thank you, a clean A/B. +9.3% decode from the PLE pool with the same text, and 37.69 tok/s with the automatic cache and the display reserve. I write the results page now, and I fix the cosmetic line (two decimals).

Before task 18, one rerun, because your profile changes my M3 reading. On ecd552f the k2 PLE rows took 4315 ms (42 ms a forward), on a PC at the RAM limit (the old 116 GB load peak). That host stall was a large part of each verify step, so M3 may have measured the RAM pressure more than the drafts. On 3f570f3 it is 403 ms.

M3 again on 3f570f3, as the user would run it: --expert-cache-gb -1 (each K gets its own automatic cache, so print the slots), no NSLM_VRAM_RESERVE_MB, code and chat prompts, --drafts 0, 1, 2, 3 in order, twice each. Report tok/s, tokens a forward, md5, slots, peak VRAM and min Available. If drafts now pass gate MC (code at least 10% faster, chat not slower), CUDA gets the same verdict as Metal.

Then task 18. About -DNSLM_G_FULL: fine for your A/B, but please drop the old path in the same branch once the A/B is done (one code path).
