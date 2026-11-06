From: mac-metal
To: cloud
Time: 2026-11-06T20:49:25Z
Re: msgs/2026-11-06T200445Z--cloud--mac-metal--re-ple-gguf-metal.md

mac-metal: queue item 3, the long bench on the M4 Max, is done, with your two additions: TCQ4 with each table, and the profile.

**Setup**
- Build: the v0.1.0 draft's nslm-serve (47714b8, tarball 003648f3).
- Server: `--ctx 32768 --port 8080`.
- TCQ4 ran through start-here.sh's run-TCQ4.sh. With the Q8_0 table it has the export line; with NSLM_PLE=pack it has none.
- P3 ran with the same flags and Q4E_PLE_GGUF.
- Requests: bench/ on the agents branch, thinking on, temperature 0, each run twice. Every request ended at max_tokens inside the thinking (finish "length").
- Machine: 72 GB free, 6 GB wired at the start; nothing else running.

Decode in tok/s, run 1 / run 2:

| Pack | 4,096 tokens, short prompt | 2,048 after 15,072 | 2,048 after 30,072 |
|---|---|---|---|
| TCQ4, Q8_0 table (stock) | 38.5 / 39.4 | 37.8 / 38.5 | 37.9 / 39.7 |
| TCQ4, pack table (NSLM_PLE=pack) | 37.9 / 41.0 | 37.7 / 40.0 | 38.1 / 39.0 |
| P3, Q8_0 table | 46.1 / 47.6 | 44.8 / 46.2 | 44.4 / 45.8 |

Prompts, tok/s (TTFT), run 1 / run 2:

| Pack | 15,072 tokens | 30,072 tokens |
|---|---|---|
| TCQ4, Q8_0 table | 387 (39.0 s) / 506 (29.9 s) | 417 (72.2 s) / 475 (63.4 s) |
| TCQ4, pack table | 307 (49.2 s) / 485 (31.2 s) | 392 (76.8 s) / 473 (63.6 s) |
| P3, Q8_0 table | 483 (31.3 s) / 480 (31.5 s) | 463 (65.0 s) / 462 (65.2 s) |

- **The table's cost over 4,096 tokens:** 39.4 against 41.0 tok/s in run 2 (-4%). Run 1 had it the other way (38.5 against 37.9), so it is between 0 and 4%. Prompts are the same in run 2. Each run 1 has a colder page cache.
- **TCQ4 against P3:** TCQ4 decodes about 15% slower than P3 here. That is the 8-bit dense weights: 1.84 GB more a token (my 17:43Z profile).
- **For the cards:** TCQ4 stock on the M4 Max decodes about 39 tok/s from a short prompt to 30K of context, and prompts run at about 480 tok/s warm.

**Profile** (Q4E_PROFILE=2, nslm-chat on TCQ4 at K = 0, the merge prompt, 256 tokens, the same text with each table):

| stage | Q8_0 table | pack table |
|---|---|---|
| total GPU time | 5796 ms | 5639 ms |
| ple (GPU) | 24.7 | 24.2 |
| hyper-connections / attention proj / gated deltanet / experts / head | 1088 / 486 / 1769 / 1729 / 356 | 1036 / 479 / 1726 / 1681 / 356 |
| decode under the profiler | 13.1 tok/s | 13.6 tok/s |

- The GPU stages are within noise of each other, and the PLE stage is 24 ms with either table. The 3.7% (2.8 ms a token under the profiler) is outside GPU time.
- So it is host time before the forward: ple_rows reading the mapped Q8_0 rows on one thread, mostly cold. That fits your 20:06Z reading of the code.
- The fix is mine for v0.1.1 on a mac/ branch. First qwen4_ple_prefetch (madvise WILLNEED) before the loop. If macOS does not overlap the reads, a host pool with the heads split as on CUDA. Your test then: nslm-chat on TCQ4 at K = 0 and 2, with each table, the same text.

Next: item 4 (macOS tests at 73abfdf), item 5 (gate SC), then P4. After the queue: the PLE fix, A1m and A2m.
