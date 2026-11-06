# Trellis experts on the M4 Max: decode and prompt against P3 (2026-11-05)

**Result.** On the M4 Max, every trellis pack decodes 1.7% to 2.3% faster than P3 seeds without drafts, and reads
prompts at P3's speed. All experts stay in unified memory, so there are no PCIe copies to save, as on the RTX 4080.
The gains on a Mac are size and quality. Trellis K = 4 has a KL of 0.0398 against P3's 0.0600 at P3's speed, for
1.9 GB more. Trellis K = 3 needs 13.2 GB less for its experts.

**Status.** Current. Gate TC of the [trellis plan](../plans/2026-11-05-trellis-experts.md) (step T3) on the M4 Max.
Read [Trellis experts on the RTX 4080](2026-11-05-trellis-speed-4080.md) for the GPU with a cache. Read
[The pages that a session uses](2026-11-05-trellis-ple-working-set-m4max.md) for the resident size of the PLE table.

## Setup

- Machine: mac-metal's M4 Max, 03:00 to 03:46 UTC. The load average was 0.8 to 2.9 and falling. The Mac ran nothing
  else.
- Build: `nslm-chat` of `main` `6f90957`, with `--no-think`, as for [MTP drafts in nslm-chat](2026-11-05-qwen38-mtp-chat-m4max.md).
- Packs: the P3 folder with each pack's `.blkt` banks beside its shards (links). The Metal encoder wrote K = 3 and K = 4,
  byte for byte the banks of the KL pages. The mix is `gate_proj` and `up_proj` from K = 3 and `down_proj` from K = 4.
- Runs: for each pack, read its files into the page cache, decode 64 tokens to warm up, then two rounds of at most 256
  tokens. The table gives round 2. Round 1 of the first block ran low.
- Prompt: 8,506 tokens (24 KB of `docs/log.md`), with the files in the page cache.
- Memory: RSS (`/usr/bin/time -l`) and the footprint (`vmmap`).

## Results

Decode, tokens per second (round 2). K is the number of MTP drafts.

| | P3 | Trellis K = 3 | Trellis mix | Trellis K = 4 |
|---|---|---|---|---|
| Code, K = 0 | 48.0 | 48.8 | 48.8 | 48.9 |
| Code, K = 2 (tokens a forward) | 70.2 (2.87) | 71.0 (2.83) | 71.1 (2.83) | 72.0 (2.87) |
| Chat, K = 0 | 47.3 | 48.1 | 48.2 | 48.4 |
| Chat, K = 2 (tokens a forward) | 47.1 (1.89) | 50.2 (1.91) | 46.4 (1.80) | 50.2 (1.98) |
| Prompt of 8,506 tokens, tokens per second (2 runs) | 496.3 / 488.9 | 506.2 / 494.8 | 502.9 / 499.8 | 492.0 |
| Expert GB | 60.4 | 47.2 | 52.2 | 62.3 |
| The pack's files, GB (experts, 32.0 of PLE, 5.8 of other weights) | 98.2 | 84.9 | 90.0 | 100.1 |
| KL against BF16 (llama.cpp) | 0.0600 | 0.0765 | 0.0619 | 0.0398 |

Each pack wrote the same text in both rounds. K = 4's code answer is P3's, byte for byte.

## Notes

- Without drafts, the trellis packs are 1.7% to 2.3% faster. The trellis gathers take 10% (gate and up) and 5% (down)
  less time than P3's, and the rest of a token is the same.
- With drafts, the speed follows each text's acceptance (1.80 to 2.87 tokens a forward), not the format.
- K = 4 read the prompt at 444 tokens per second when it ran after K = 3 and the mix. Its 100 GB of files did not all
  stay in the page cache with the other banks cached too. On a 128 GB Mac, K = 4 is at the edge of the page cache for
  prompts that use every expert. A P3 prompt run straight after its own decode block (233 tokens per second) had the
  same cause. The runs above replace it.
- RSS is 0.9 to 1.2 GB, and the footprint is 4.0 GB (3.6 GB of Metal buffers) for each pack. The weights are clean
  pages of the mapped files, so a Mac needs a page cache that holds the files that a run uses.
- A 64 GB Mac and K = 3: the files without the PLE table are 53 GB. A token reads only a few rows of the table, but its
  resident size is not measured yet.
