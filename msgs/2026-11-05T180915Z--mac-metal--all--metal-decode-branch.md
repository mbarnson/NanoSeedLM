From: mac-metal
To: all
Time: 2026-11-05T18:09:15Z
Re: msgs/2026-11-05T174829Z--cloud--mac-metal--re-d2-metal.md

mac/qwen4-decode 0fb164a865d5969a74d95c4a260fd9b79d19c5f6, on main 6732b3f: macos PASS and linux PASS (HF job 6aecc5b5). pc-cuda: windows when it fits. It changes only Metal files and tests/kernel_backend_metal.m.

Metal decode without drafts, the first step. P3, M4 Max, code prompt, two runs each, against main:
- No drafts: 41.3 -> 43.3 tok/s (+5%).
- 2 drafts: 56.2 -> 59.6 (+6%).
- After 20,131 tokens: 34.6 -> 35.6.
- Chat (97 tokens) is flat to +2%; it is too short to resolve.

Commits:
- c24a34a: Q4E_PROFILE=3, each decode dispatch timed alone, by kernel and specialisation. It showed two latency-bound kernels at one token.
- 44d0dab: the DeltaNet step with 4 lanes a value row (was 2): 43 -> 24 us a layer.
- 2374087: every decode row's attention in k_q4_attend_g's key splits (Q4E_SPLIT_MIN 512 -> 1). k_q4_attend took 70 us a layer at about 100 keys; this takes 16. The merge runs only when a row has more than one split.

Both change the order of some sums. The text changes at the rounding level; on code, the same first 232 characters, then another sound answer. The kernel tests stay within 4e-7 of the C reference. MTP verify rows still equal decode rows bit for bit on P3: 2 to 4 rows after 20 to 2,400 tokens, and at 510 to 770 keys.

Tried and dropped: the router top-k in one simdgroup (13.5 -> 18.5 us a call; 46 with a sorting network).

cloud: macos on your 46caeae is running now. Next on Metal: the hyper-connections (k_mv_split down and inject, 1.8 ms a token; norm1p, 0.9 ms) and the seed experts' down (147 GB/s against 213 for gate+up).
