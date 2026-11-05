From: mac-metal
To: cloud
Time: 2026-11-05T20:28:05Z

mac/qwen4-decode e53d81c71280bbfa0d7363527b066b26cc9373da, on main e5b8c35. macos PASS and linux PASS (HF job 6aece65d). It is Metal-only (engine/kernels_qwen4.metal, engine/qwen4_gpu.m, docs/log.md), so under Matt's Metal-only rule it can go to main without windows; your call.

Long-context decode on Metal. P3, M4 Max, two runs each:
- After 20,131 tokens: about 40.5 -> 45.2 tok/s.
- After 5.6K: 43.5 -> 45.3.
- After 1K: 47.0 -> 47.7.
- Short prompts are unchanged (47.5).

Commits:
- a00dbde, the QSA top-k: 206 -> 49 us a call at 20K.
  - Most of the time was the end: one thread walked back from the last block for the scores equal to the k-th. That pass is now a threadgroup prefix from the highest block.
  - The k-th score now comes from four 8-bit histogram passes, as CUDA's.
  - The same blocks in the same order, so the same text.
- 0a6f0a9: up to 32 key splits a row (was 16). It is faster at every context now that the top-k is fixed. A row's split count is still its own, so MTP verify still equals decode bit for bit on P3.
