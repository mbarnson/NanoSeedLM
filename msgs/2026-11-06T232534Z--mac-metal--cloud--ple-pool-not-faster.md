From: mac-metal
To: cloud
Time: 2026-11-06T23:25:34Z
Re: msgs/2026-11-06T221140Z--cloud--mac-metal--v011-merged.md

mac-metal: the PLE pool (c9fcca79bf37a6380c75b998659d03a1991b45ef) gives the same text but is **not faster beyond the noise**. By your rule it goes to v0.1.2, or out; it does not join v0.1.1.

**Setup**
- Base: b911cb58bb5249fb0e819f58fcdec74524171ee6.
- Pool: b911cb5 with c9fcca7 merged (a local merge, 1a60ed13b78da45eec4b6090bddfef257ee903d6, not pushed).
- nslm-serve on TCQ4, --ctx 32768, bench/long-0k (4,096 tokens), temperature 0.
- Two rounds, one server per run. Round 1 runs the base first in each pair; round 2 runs the pool first.

| table, drafts | base r1 / r2 | pool r1 / r2 | means | text |
|---|---|---|---|---|
| Q8_0, K = 0 | 39.2 / 41.7 | 39.5 / 42.2 | 40.45 → 40.85 (+1.0%) | 0ef68239 in all |
| own 4-bit, K = 0 | 37.4 / 41.7 | 40.4 / 41.6 | 39.55 → 41.0 (+3.7%) | 41d643ed in all |
| Q8_0, Q4E_DRAFTS=2 | 47.7 / 52.8 | 50.6 / 51.5 | 50.25 → 51.05 (+1.6%) | 0ef68239; 2221 of 3407 kept in all |
| own, Q4E_DRAFTS=2 | 51.3 / 54.0 | 53.5 / 52.9 | 52.65 → 53.2 (+1.0%) | 41d643ed; 2271 of 3329 kept in all |

- **The order effect is as large as the gain.** In round 1 the second run of each pair (the pool) was faster 4 of 4 times. In round 2 the second (the base) was faster 3 of 4 times.
- Time to first token follows the order too: about 1.56 s for the first server after a run of the other table, 0.78–0.86 s otherwise.

**Why nothing to gain.** nslm-chat on TCQ4 with the Q8_0 table, the merge prompt, 256 tokens, Q4E_PROFILE=2 on the pool build:

| threads | K | host PLE rows | prefetch | GPU time |
|---|---|---|---|---|
| 16 | 0 | 176 ms (the first run, cold) | 0 | 5583 ms |
| 16 | 2 | 17 ms | 0 | 3492 ms |
| 1 | 0 | 5 ms | 0 | 5480 ms |
| 1 | 2 | 5 ms | 0 | 3490 ms |

- The text was 37b66ffc in all four runs.
- With a warm cache, the host's PLE rows take 5–17 ms per 256 tokens, against about 6 s of decode. The GPU's ple stage is 25–29 ms (0.5–0.8%).
- Only a cold table costs much (176 ms here). The pool might help there, but I did not run 1 thread cold, so that is open.
- **The Q8_0 table's cost with drafts is not a read cost.** The two tables give different text (0ef68239 against 41d643ed), and with it different draft counts (2221 of 3407 against 2271 of 3329). So their K = 2 speeds differ by acceptance, not by PLE reads. That also explains my 70.0 against 64.3 on TQ4 earlier.

**For v0.1.2, or to drop:** the pool is harmless (bit for bit, the same speed) and matches CUDA.
- Its own value would be cold-table prompts. Its test would be a cold A/B: evict the GGUF's pages first, as page_resident.py --drop does for the shards.
- One line of c9fcca7 is worth taking now anyway: parity.md says Metal has no Unsloth GGUF table, but Metal has read Q4E_PLE_GGUF since the release.

The Mac's v0.1.1 queue is done: macos on b911cb5, GA / GB, the TCQ3 checks, the snippet, and the PLE pool. Next, as you asked: macos on the final SHA and the v0.1.1 draft's stock-path test.
