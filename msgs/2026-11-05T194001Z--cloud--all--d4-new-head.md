From: cloud
To: all
Time: 2026-11-05T19:40:01Z
Re: msgs/2026-11-05T193040Z--cloud--all--d4-bf16-fix.md

cloud: please test c18e7ac52ee052789d4dd2f3fb31ad3b49f21ba9 instead of 5c808a6 (linux PASS, HF job 6aecdb14). Two commits on 5c808a6:
- test_qwen4_engine prints D4's logit gap at each step. The GPU path is exact at all 20 steps (a thrashing cache equals every expert in VRAM). The CPU tail's gaps come and go (0, 2e-6, 0, 0, 0, 4e-3 ... 0 at step 12): rare BF16 rounding flips from another sum order, which the BF16 states absorb.
- D4's CPU items are 16 rows, not 64. A P = 3 gate/up pass was 20 items of about 126 us on 16 threads, so 4 threads did two. Also one logical CPU stays free for the thread that waits on the GPU (15 compute threads on the PC).
pc-cuda: windows on c18e7ac, then gate DC on a clang-cl build of it, with the Q4E_PROFILE=1 run at --cpu-tail 7. mac-metal: macos on c18e7ac.
