# Long decode and long prompts on the M4 Max (2026-11-06)

**Result.** On the stock path of v0.1.0, the M4 Max decodes 4,096 tokens of TCQ4 at about 39 tokens per second. After
a prompt of 30,000 tokens it still decodes at about 39. Warm prompts run at about 480 tokens per second. Unsloth's Q8_0
PLE table costs 0% to 4% of decode against the pack's own table: the host reads its rows on one thread.

**Status.** Current. The Mac's part of the long runs for the cards. The PC's part is
[the RTX 4080 page](2026-11-06-long-bench-4080.md). The note on the cost of the Q8_0 table is replaced by
[the PLE rows on a host pool](2026-11-06-ple-pool-m4max.md): the host reads take 5 to 17 ms for 256 tokens.

## Setup

- Machine: M4 Max with 128 GB, macOS. Nothing else ran. 72 GB free and 6 GB wired at the start.
- Engine: the v0.1.0 tarball (`003648f3`, built from `47714b8`), `nslm-serve --ctx 32768 --port 8080`.
- TCQ4 ran through `start-here.sh`'s `run-TCQ4.sh`, with the Q8_0 table, then with `NSLM_PLE=pack` (the pack's 4-bit
  table). P3 ran with the same flags and `Q4E_PLE_GGUF` set to the Q8_0 table.
- Requests: `bench/` on the `agents` branch, as on the RTX 4080. Thinking on, temperature 0, no MTP drafts. Each
  request ran twice, and each ended at `max_tokens` inside the thinking.
- Profile: `Q4E_PROFILE=2`, `nslm-chat` on TCQ4, 256 tokens, the same text with each table.

## Results

Decode in tokens per second (run 1 / run 2):

| Pack | 4,096 tokens, short prompt | 2,048 tokens after 15,072 | 2,048 tokens after 30,072 |
|---|---|---|---|
| TCQ4, Q8_0 table (stock) | 38.5 / 39.4 | 37.8 / 38.5 | 37.9 / 39.7 |
| TCQ4, pack table | 37.9 / 41.0 | 37.7 / 40.0 | 38.1 / 39.0 |
| P3, Q8_0 table | 46.1 / 47.6 | 44.8 / 46.2 | 44.4 / 45.8 |

Prompts in tokens per second, with the time to the first token (run 1 / run 2):

| Pack | 15,072 tokens | 30,072 tokens |
|---|---|---|
| TCQ4, Q8_0 table | 387 (39.0 s) / 506 (29.9 s) | 417 (72.2 s) / 475 (63.4 s) |
| TCQ4, pack table | 307 (49.2 s) / 485 (31.2 s) | 392 (76.8 s) / 473 (63.6 s) |
| P3, Q8_0 table | 483 (31.3 s) / 480 (31.5 s) | 463 (65.0 s) / 462 (65.2 s) |

The profile (GPU time of 256 tokens, ms):

| Stage | Q8_0 table | Pack table |
|---|---|---|
| Total | 5,796 | 5,639 |
| PLE | 24.7 | 24.2 |
| Hyper-connections / attention projections / DeltaNet / experts / head | 1,088 / 486 / 1,769 / 1,729 / 356 | 1,036 / 479 / 1,726 / 1,681 / 356 |

## Notes

- The Q8_0 table's cost is outside GPU time. The GPU stages are the same within noise, and the PLE stage is 24 ms with
  each table. The host reads the mapped Q8_0 rows on one thread before each forward, and most of them are cold.
  mac-metal plans the fix for v0.1.1. The engine can ask for the rows' pages first (`qwen4_ple_prefetch`). Or it can
  read the heads on a pool, as the CUDA engine does.
- TCQ4 decodes about 15% slower than P3 on this Mac. Its 8-bit dense weights read 1.84 GB more for each token.
- Each run 1 had a colder page cache, so its prompts are slower.
