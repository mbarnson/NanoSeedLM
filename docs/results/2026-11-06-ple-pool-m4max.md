# The PLE rows on a host pool on the M4 Max (2026-11-06)

**Result.** A pool of host threads for the PLE rows, as in the CUDA engine, gives the same text on Metal. Its speed is
within the noise: +1.0% to +3.7%, and the order of the runs moves decode by as much. With a warm page cache, the host
reads the PLE rows of 256 tokens in 5 to 17 ms, against about 6 s of decode. The Q8_0 table's speed difference with
drafts comes from its other text and draft counts, not from row reads.

**Status.** Current. The pool (`mac/qwen4-ple-pool` `c9fcca7`) does not join v0.1.1. A cold-table test may still
justify it.

## Setup

- Mac: M4 Max with 128 GB, macOS. Nothing else ran.
- Base: `b911cb5`. Pool: `b911cb5` with `c9fcca7` merged (a local merge).
- `nslm-serve` on TCQ4, `--ctx 32768`, `bench/long-0k.json` (4,096 tokens), temperature 0. Two rounds, one server for
  each run. Round 1 ran the base first in each pair, and round 2 the pool first.
- Profile: `nslm-chat` on TCQ4 with the Q8_0 table, 256 tokens, `Q4E_PROFILE=2`, 16 threads and 1.

## Results

Decode in tokens per second (round 1 / round 2):

| Table, drafts | Base | Pool | Means | Text |
|---|---|---|---|---|
| Q8_0, no drafts | 39.2 / 41.7 | 39.5 / 42.2 | 40.45 to 40.85 (+1.0%) | the same |
| The pack's 4-bit, no drafts | 37.4 / 41.7 | 40.4 / 41.6 | 39.55 to 41.0 (+3.7%) | the same |
| Q8_0, 2 drafts | 47.7 / 52.8 | 50.6 / 51.5 | 50.25 to 51.05 (+1.6%) | the same, 2221 of 3407 kept |
| The pack's 4-bit, 2 drafts | 51.3 / 54.0 | 53.5 / 52.9 | 52.65 to 53.2 (+1.0%) | the same, 2271 of 3329 kept |

Host time of the PLE rows in 256 tokens (pool build):

| Threads | Drafts | PLE rows | GPU time |
|---|---|---|---|
| 16 | 0 | 176 ms (the first run, a cold table) | 5,583 ms |
| 16 | 2 | 17 ms | 3,492 ms |
| 1 | 0 | 5 ms | 5,480 ms |
| 1 | 2 | 5 ms | 3,490 ms |

## Notes

- The second run of a pair was faster in 7 of 8 pairs. So the order of the runs is as large as the pool's gain.
- The two tables give different text, so they keep different shares of the drafts. That, not row reads, is why the Q8_0
  table was slower with drafts in [the release download](2026-11-06-release-downloads-m4max.md) and
  [the long runs](2026-11-06-long-bench-m4max.md).
- Only a cold table costs much. A test for the pool would evict the table's pages first, as `tools/page_resident.py
  --drop` does for the shards.
