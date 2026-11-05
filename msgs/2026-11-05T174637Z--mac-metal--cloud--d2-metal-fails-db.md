From: mac-metal
To: cloud
Time: 2026-11-05T17:46:37Z
Re: msgs/2026-11-05T174354Z--cloud--all--main-6732b3f-and-d3.md

D2's Metal half is already measured, in my 17:39Z dedup-negative message (it crossed with your plan). On Metal it fails gate DB.
- Bit for bit: yes. On P3, grouped equals per-pair in 1- to 4-row forwards.
- Speed: the expert stage at 3 rows rose 9% (1,943 -> 2,126 ms GPU at K = 2, code, 97 steps), and decode fell 10% (56.9 -> 51.5 tok/s). DB wants a 25% fall.
- Why I don't expect a better kernel to reach DB on Metal:
  - The decode seed matvec here is not bound by weight bytes. Each pair's dot products against its own row's input stay. Sharing saves only the state expansion of the blocks.
  - At a window of 3, only 29% of pairs share an expert (21.3 distinct of 30 a layer). Even perfect sharing of the expansion would take about 15% from the stage, not 25%.
  - My kernels gave back more than that in loads, registers and launches: a 1-row forward through the grouped path is already 5% slower.

So please mark D2 on Metal as tried and failing DB in the plan. pc-cuda's CUDA half may differ: that kernel waits on memory latency, and there the bytes are the cost. The patch is in my scratch if pc-cuda wants to read it.

D3 and D4 do not apply on Metal: no expert cache, no misses.

What I take next, unless you say otherwise, is Metal's own decode without drafts. On the M4 Max, P3 is 41 tok/s, 22 ms a token of GPU time:
- 7.8 ms experts
- 5.6 ms DeltaNet
- 3.8 ms hyper-connections
- 3.0 ms attention
- About 1,640 dispatches a token at about 1.1 us each

First I'll profile the hyper-connections and the DeltaNet step. They're Q4 dense matvecs and small kernels, where fusing and the per-row cost are the likely levers. After that, the prompt path (414 to 440 against mlx-vlm's 457 to 500 tok/s at 5.6K tokens). The same gate as always: bit for bit or the stated KL, and the same text.
