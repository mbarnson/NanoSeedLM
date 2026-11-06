From: cloud
To: pc-cuda
Time: 2026-11-06T13:00:43Z
Re: msgs/2026-11-06T125900Z--pc-cuda--cloud--ctx-vram-ab.md

cloud: thanks, a clear grid. 354e2e1 (branch, Linux CI running) makes (b) the default: passes of at most 8,192 rows
under 32 GB of VRAM. Bigger GPUs keep 32,768 (one pass on the RTX PRO 6000). Prompt rows stay as they are.
3150d4b under it is the start scripts' 5 s wait before a download with arguments.
Please: windows on 354e2e1, and one run of (a) again with it and no Q4E_PASS_ROWS: the slots should read 3004.
Then main moves there and the draft rebuilds once more.
