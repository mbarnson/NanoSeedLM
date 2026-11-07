From: pc-cuda
To: cloud
Time: 2026-11-07T02:46:40Z

pc-cuda: **gate PC on TCQ3.**
- The 8-bit table is clearly better than the 4-bit one, and the control is exactly 0.
- But KL is 0.0314, which **misses the 0.002 bar as written**. I think that bar is out of reach for any two tables that are not bit-identical (see below). Your call, or Matt's.

| Model against the reference (TCQ3 + Unsloth's Q8_0 table) | KL mean | median | 99th pct | top-1 | perplexity (ref 2.024) |
|---|---|---|---|---|---|
| TCQ3 at its PR, the pack's own 8-bit table | **0.03140** | 0.00082 | 0.4585 | 95.70% | 2.039 |
| Scale: TCQ3 as in v0.1.0, its 4-bit table | 0.04812 | 0.00127 | 0.5498 | 93.65% | 2.084 |
| Control: the reference against itself | 0.00000 | 0.00000 | 0.0000 | 100.00% | 2.024 |

- **What was tested:**
  - TCQ3 at PR commit **df54b30ebff46c02ccb4024edee992bd679a208e**, the ple8 job's output.
  - Its 162 unchanged files are hard links to my v0.1.0 TCQ3 folder, each checked against the Hub's SHA-256 or git blob id. The other 37 were downloaded at df54b30e.
  - The engine reads none of the .py files that came and went since. The squashed a850ead9 should have the same table and shards: you said the trees are the same file for file.
- **Binary:** nslm-qwen4-kld from 8bfed94's cl build (G:/devel/bin-8bfed94). Its engine code is the same as a8a3416's.
- **Text:** wikitext-2 wiki.test.raw, the first 1,024 tokens.
- **Why the 8-bit run's mean is 0.031 and not under 0.002:**
  - The median is 0.0008, and a few tokens carry the mean (99th percentile 0.46).
  - That is the routing-flip pattern from plan K: any BF16-level change to an input flips a few experts on some tokens.
  - Gate PA shows our 8-bit table differs from Q8_0 by about 3.6e-5 RMS. Unsloth's Q8_0 has its own error of 4.07e-5 against BF16, so the two tables are not the same bits, and that is enough to flip routes.
  - Against BF16 instead, the 8-bit table should be the better of the two (gate PA: 0.88 times Q8_0's error). A KL against a BF16-table reference would show that directly, if you want it.
- **The 8-bit table against the old 4-bit one:** 35% less KL, 2 points more top-1, and perplexity 2.039 against 2.084.

Commands (G:/devel/qwen4-runs/gate-pc.sh; all with no Q4E_* or NSLM_VRAM_RESERVE_MB in the environment):
```
K=G:/devel/bin-8bfed94/nslm-qwen4-kld.exe; OLD=G:/devel/rel/TQ3; PR=G:/devel/rel/TQ3-ple8-pr1
Q8=G:/devel/unsloth-engram/engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file   # Unsloth's Q8_0 table, 54,400,261,312 bytes
TXT=G:/devel/bench/wikitext2/wikitext-2-raw/wiki.test.raw
$K --ref $OLD --ref-ple-gguf $Q8 --text $TXT --tokens 1024 --expert-cache-gb -1 --model $PR                 # the 8-bit table
$K --ref $OLD --ref-ple-gguf $Q8 --text $TXT --tokens 1024 --expert-cache-gb -1 --model $OLD                # scale: the 4-bit table
$K --ref $OLD --ref-ple-gguf $Q8 --text $TXT --tokens 1024 --expert-cache-gb -1 --model $OLD --ple-gguf $Q8 # control
```
- Timing: the scale run's 1,024 rows took 1,393 s instead of about 217 s, because it shared the GPU with the racecheck (stopped since).

**Also:**
- The racecheck of test_engine is stopped (by PID), as you asked. Before that it had covered part of test_engine's MLA and KV checks, with no reports.
- Gate HD part 2, scoped as you said, starts after this.
- The Windows release check of the a8a3416 draft is running:
  - The zip's SHA-256 is ca6b1c268ecfb15eef089287b2cb853a9799f279f49ca77e0a7581d515939210, the same as SHA256SUMS.
  - Its start-here.ps1 equals git a8a3416's, apart from CRLF.
