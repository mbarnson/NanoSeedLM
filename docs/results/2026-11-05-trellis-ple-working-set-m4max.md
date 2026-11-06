# Trellis K = 3 on the M4 Max: the pages that a session uses (2026-11-05)

**Result.** A session of about 13,000 tokens with trellis K = 3 keeps about 55 GB of the pack's files in the page
cache. That is 47.2 GB of trellis banks, 3.1 GB of other tensors and 5.0 GB of the 32.0 GB PLE table. The PLE
part grows by about 0.5 MB for each generated token and 0.3 MB for each prompt token. With a cold PLE table, decode
is about 10% slower while its rows come from the SSD. So K = 3 may fit a 64 GB Mac, but no 64 GB Mac is measured.

**Status.** Current. It answers a note of [Trellis experts on the M4 Max](2026-11-05-trellis-speed-m4max.md).

## Setup

- Machine: mac-metal's M4 Max (128 GB). The Mac ran nothing else, with a load average of about 2.
- Build: `nslm-chat` of `main` `b301be5`, the trellis K = 3 pack (L = 12) of the M4 Max page.
- Tool: `tools/page_resident.py` (`39afe70`). It calls `mincore` over the byte range of each tensor in the shards and
  over the `.blkt` banks, with pages of 16 KB. A page counts only its overlap with each tensor.
- Start: `--drop` cleared every shard range from the page cache (`msync` with `MS_INVALIDATE` on a read-only map). So
  the PLE table, the P3 seed experts and the other weights read 0.00 GB. The banks stayed in the page cache (47.19 GB).
- Runs, one after the other, with the resident sizes read after each:
  1. The code and chat prompts with thinking on (EOS at 242 and 95 tokens).
  2. A B-tree explainer with `--no-think`, 2,000 tokens.
  3. A lighthouse story with `--no-think`, 2,000 tokens.
  4. A prompt of 8,506 tokens (24 KB of `docs/log.md`).

```bash
python3 tools/page_resident.py PACK_DIR --drop
nslm-chat ...   # each run of the list above
python3 tools/page_resident.py PACK_DIR
```

## Results

The PLE table after each run:

| After | Tokens (prompt + generated) | PLE resident, of 32.02 GB |
|---|---|---|
| Code and chat | 130 + 337 | 0.28 GB |
| The B-tree explainer | + 38 + 2,000 | 1.28 GB |
| The lighthouse story | + 36 + 2,000 | 2.46 GB |
| The prompt of 8,506 tokens | + 8,506 | 4.97 GB |

Each kind after the last run, from the tool:

| Kind | Resident | Of |
|---|---|---|
| `.blkt` banks (trellis K = 3) | 47.19 GB | 47.19 GB |
| The PLE table | 4.97 GB | 32.02 GB |
| Other tensors | 3.10 GB | 4.00 GB |
| The MTP head | 0.00 GB | 1.73 GB |
| Routed experts in the shards (P3 seeds) | 0.00 GB | 60.40 GB |

The banks replace the P3 seed experts, so the engine does not read them. Decode in these runs was 41.8 to 43.1 tokens per
second, against 48.1 to 48.8 with a warm page cache on the M4 Max page.

## Notes

- Each token reads a few rows of the PLE table, at scattered places. So each row costs a full 16 KB page. 13,000
  tokens of varied text touched 16% of the table.
- Of the other tensors, 3.10 of 4.00 GB are resident. The embedding and the output head are read only in part. The
  MTP head is not read without drafts.
- A first script counted a page once for each tensor on it. It gave 3.13 GB of 5.73 GB for the other tensors and the
  head together. The tool corrects this. The PLE numbers do not change.
- For a 64 GB Mac: about 50 GB stays fixed (the banks and the other tensors), and macOS needs about 8 to 10 GB. That
  leaves a few GB for the hot PLE pages. A long session then reads cold PLE rows from the SSD again. This is an
  estimate from the 128 GB Mac.
- The PLE pages are clean pages of a mapped file. So macOS can drop cold rows and read them again later.
