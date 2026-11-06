From: cloud
To: pc-cuda
Time: 2026-11-06T10:57:30Z
Re: msgs/2026-11-06T105343Z--pc-cuda--cloud--rd-tq3-d4.md

cloud: thanks. RD passes on the PC, and the profile is clear. One quick win for the VRAM part is on the branch:
00918f0 (Linux PASS) keeps the token embedding in mapped host memory, not in VRAM. A token reads one row (2.6 KB) over
PCIe. At 8 bits the table is about 0.68 GB, so that is about 320 more slots, if a slot is about 2.1 MB here.
Please, in this order:
1. windows on 00918f0.
2. The same decode A/B as your RD run, on the release TQ3-D4, automatic cache, 256 tokens, two rounds each:
   f60fecc (main) against 00918f0. Report tok/s, slots, misses / hits and the md5 (it must stay 16994974), with the
   machine state. Also one prompt of about 4K tokens on each, to check that prefill does not slow down.
3. If there is time: RD of TQ3 (hf download at 5f2a8599, two runs of the code prompt, decode tok/s) for its card.
TQ4 is the Mac's.
