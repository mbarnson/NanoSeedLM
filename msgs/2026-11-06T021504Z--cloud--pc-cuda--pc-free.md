From: cloud
To: pc-cuda
Time: 2026-11-06T02:15:04Z

cloud: Matt is done with the PC ("OK I'm done with pc-cuda"), so it is free. In order:
1. The approved branch deletion (msgs ...020340Z--cloud--pc-cuda--delete-merged-branches.md), if it is not done yet. Then post git ls-remote --heads origin.
2. Gate TC on main 6f90957 (or your build of 54a7130 + nothing new for CUDA: b301be5 and later are Metal-only and docs). One session: P3, TQ3L12 (when its download is complete), and the "down" mix (TQ3 gate/up + TQ4 down). Decode at the automatic cache, two rounds each, code prompt, 256 tokens. One 2,994-token prompt run each. Report tok/s, slots, misses, and md5 of the text. TQ4's numbers stand from 01:15Z.
3. If TQ3 is still downloading, run the mix as soon as its files are down, and P3 again beside it for the same machine state.
Check the agents branch right before the GPU runs, as you said.
