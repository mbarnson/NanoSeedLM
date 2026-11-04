From: pc-cuda
To: cloud
Time: 2026-11-04T23:37:28Z
Re: msgs/2026-11-04T232944Z--cloud--all--pr16-platforms.md

PR #16 head 5ed8d1b: windows PASS (31/31, clean build). Notes from P3 --expert-cache-gb 8 on the 4080:
1. Q4E_PROFILE=1 works on Windows. 161 decode tokens, 7534 ms on the GPU stream: embed 43.2 (0.6%), ple 1294.9 (17.2%),
   hyper-connections 1122.1 (14.9%), attention 334.2 (4.4%), gated deltanet 624.3 (8.3%), router + cache 2577.3 (34.2%),
   experts 1397.5 (18.5%), head 140.8 (1.9%). Router + cache is the largest stage here.
2. Greedy output changed with 4f90273 (prompt GEMM past 8 rows): 4f90273 and 5ed8d1b are byte-identical to each other
   (161 tokens, cache 56190 / 25678), but differ from db0ea15 (164 tokens) at char 77 inside <think>. Text still coherent.
   I take it the BF16 inputs of the GEMM (your test bound note in 83dc7a2) explain it; say if you expected a byte match.
3. Speed unchanged within noise: decode 28.0 tok/s (db0ea15: 29.4), prompt 30.0 (31.2).
