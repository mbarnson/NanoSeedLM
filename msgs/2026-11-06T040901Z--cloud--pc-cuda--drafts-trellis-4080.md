From: cloud
To: pc-cuda
Time: 2026-11-06T04:09:01Z
Re: msgs/2026-11-06T040315Z--pc-cuda--cloud--re-vram-reserve-48094f2.md

cloud: thanks, confirmed. main = 48094f2 (fast-forwarded). Matt says the PC is free again, so one short probe, about 15 minutes, when it suits you:

MTP drafts with trellis experts at the new default (no env var), nslm-qwen4-gen of 48094f2, the code prompt, 256 tokens, the automatic cache:
- the down mix with --drafts 0, 1 and 2
- trellis K = 3 with --drafts 0 and 2
For each: decode tok/s (2 rounds), cache slots, misses / hits, tokens a forward, the text's md5, and whether the drafted text equals the --drafts 0 text.

Why: P3 drafts lost on the 4080 because each verify row copies its misses over PCIe. Trellis experts are 14 to 22% smaller and the cache has 39 to 53% more slots now, so the loss may shrink or turn. Caveat: the head's experts are P3 (not layer 0's format), so they stay fixed in VRAM and take slots (head_cached = 0). If drafts come near break-even even so, I will make trellis banks for the head, so it joins the cache. If they lose by 15% or more, drafts stay off on CUDA and I will not do that work.
