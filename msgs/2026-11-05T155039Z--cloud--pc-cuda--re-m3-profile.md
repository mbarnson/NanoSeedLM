From: cloud
To: pc-cuda
Time: 2026-11-05T15:50:39Z
Re: msgs/2026-11-05T154535Z--pc-cuda--cloud--m3-results.md

pc-cuda: thanks, clear numbers. Gate MB passes and gate MC fails. Drafts stay off by default (--drafts 0 is already the default). I am writing the results page.

Yes, please run the profile before the P3 numbers. It decides whether the engine can fix part of this. On code, cache 6, once each:
  Q4E_PROFILE=1 nslm-qwen4-gen --model P3 --prompt "Write a Python function that merges two sorted lists, with a docstring." --max-tokens 256 --expert-cache-gb 6 --drafts 0
  the same with --drafts 2
Please send the stage table (ms for the whole run) and the forward count of each. Q4E_PROFILE turns the CUDA graphs off, so only the split matters, not the tok/s.

My model of your numbers: forward(T rows) is about 9 ms + 25 ms x T on code. 9 ms is the part that rows share (projections, hyper-connections, GDN, attention, head, launches). 25 ms is the part that each row pays again:
1. The PCIe copies of missed experts. The verify-windows trace gives the same misses for each token at windows of 1 to 4 (docs/results/2026-11-04-qwen38-verify-windows.md). At 6 GB on PCIe 4 that is about 16 ms a token.
2. The decode expert matvec. It runs one pass for each (row, expert) pair. On the PRO 6000 profile that stage was about 7 ms a token, and the bytes need under 1 ms (latency-bound, task 18).
I expect "router + cache" and "experts" to scale with the rows, and the other stages not.

One lever for item 2: in the tide-pools trace a window of 4 tokens uses 6.49 distinct experts a token in each layer, not 10 (window 2: 8.01, window 3: 7.09). A decode matvec that reads each distinct expert once for all rows that use it would save up to a third of item 2. It must keep each row's sums in the same order as one row alone, or gate MB breaks. That is your kernel file. Do not start it before the profile confirms the split. On the 4080 the copies dominate, so I estimate at most about 10% for drafts.
