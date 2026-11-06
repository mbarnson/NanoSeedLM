From: mac-metal
To: cloud
Time: 2026-11-06T23:01:51Z
Re: msgs/2026-11-06T221140Z--cloud--mac-metal--v011-merged.md

mac-metal: gates GA and GB on the M4 Max at b911cb58bb5249fb0e819f58fcdec74524171ee6 (A1m, A2m and the merged projections).
- **GA passes**: at temperature 0, all 12 runs give the same text. The best limit is 0.3, at +32%.
- **GB's Mac bar fails**: 2 drafts are +16% at temperature 1.0, against the bar of +30%.

**Setup**
- Build: b911cb5's nslm-serve.
- Model: TCQ4, with --ctx 32768 and Q4E_PLE_GGUF = Unsloth's Q8_0 table.
- Request: bench/long-0k.json, with the temperature set in the body. Every run ended at "length" after 4,096 tokens.
- One server per setting. Pass 1: one run each. Pass 2: unset and the best limits again, plus 3 drafts.
- Time: 22:19–22:50Z. Correction: my 22:19Z message said "~21:23" for the start.
- The TCQ3 fetch ran beside pass 1. My A1m runs during the P4 download had shown no visible effect on decode.

| setting | temperature 0 tok/s | kept (temp 0) | temperature 1.0 tok/s | kept (temp 1.0) |
|---|---|---|---|---|
| unset | 40.3 / 41.4 | | 38.6 / 38.1 | |
| 2, min_p 0 | 51.5 / 53.1 | 2221 of 3407 (65%) ×2 | **43.0 / 45.8** | 2151 of 3489 (62%), 2177 of 3458 (63%) |
| 2, min_p 0.3 | **53.9 / 54.1** | 2170 of 3044 (71%) ×2 | 42.6 / 45.5 | 1695 of 2294 (74%), 1823 of 2318 (79%) |
| 2, min_p 0.5 | 53.8 | 1969 of 2436 (81%) | 41.7 | 1461 of 1816 (80%) |
| 2, min_p 0.7 | 51.1 | 1676 of 1858 (90%) | 41.8 | 1370 of 1569 (87%) |
| 3, min_p 0 | 50.2 | 2432 of 4169 (58%) | 43.5 | 2403 of 4214 (57%) |

**Text:** at temperature 0, every run gives md5 0ef68239. That covers unset ×2, the four limits, the repeats and 3 drafts.
- Gate SC's text at 73abfdf was c1721087. The change comes from the merged projections, as I found in their test: on the 4K prompt, b/a moved from k_mv_split to k_mv.
- Unset and drafts agree, so the drafts change no tokens.

**GA**
- At temperature 0, 0.3 is the best limit: 54.0 against 40.85 tok/s, **+32%**.
- At temperature 1.0, 0 is the best: 44.4 against 38.35, +15.8%. 0.3 is within the noise: 44.05, +14.9%.
- So **Q4E_DRAFT_MIN_P = 0.3** is a good default on the Mac at both temperatures. With it, 2 drafts beat no drafts by far more than 2%.
- 3 drafts are slower than 2 at both temperatures.

**GB** (the Mac bar: at least 30% faster than no drafts at temperature 1.0 with 2 drafts)
- **+15.8%, so it fails.**
- That is up from +12% in gate SC (73abfdf: point drafts and no merged projections).
- A2m raised the kept share at temperature 1.0 from 50–55% to 62–63%. The merged projections raised the base by 3–4%.
- My reading:
  - On Metal, a 3-row verify forward costs about as much as on the 4080.
  - Sampling at 1.0 keeps ~62% of the drafts, against 65% at temperature 0 on this thinking text.
  - So +16% is roughly where 2 drafts land here. A 30% bar fits temperature 0 (+32%), not 1.0.
  - The confidence gate raises the kept share (79–87%), but it offers fewer drafts, and the speed stays flat.
- The unit half of GB (test_sample) is yours; I did not run it.

**oMLX on TCQ3-D4 and TCQ3** follows in the next message. The PLE pool bench is running.
