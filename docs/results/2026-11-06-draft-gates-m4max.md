# MTP drafts in nslm-serve on the M4 Max: gates SC, GA and GB (2026-11-06)

**Result.** On the M4 Max, 2 drafts with `Q4E_DRAFT_MIN_P=0.3` make `nslm-serve` decode TCQ4 32% faster at temperature
0, with the same text in each run. At temperature 1.0 they are 16% faster. Drafts sampled from the head's distribution
(A2m) keep 62% of the drafts at temperature 1.0, against 50% to 55% before. The plans' speed bars (40% for SC, 30% at
temperature 1.0 for GB) fail.

**Status.** Current. The M4 Max's part of the gates. The RTX 4080's part is
[MTP drafts in nslm-serve on the RTX 4080](2026-11-06-draft-gates-4080.md).

## Setup

- Mac: M4 Max with 128 GB, macOS. Nothing else ran.
- Engine: `nslm-serve` of `claude/sleepy-brown-k67z9u`. Gate SC used `73abfdf` (point drafts). Gates GA and GB used
  `b911cb5` (A1m, A2m and the Metal merged projections).
- Pack: TCQ4, `Q4E_PLE_GGUF` set to Unsloth's Q8_0 table, `--ctx 32768`, one server for each setting.
- Request: `bench/long-0k.json` (4,096 tokens, thinking on), the temperature set in the request.

## Results

Gate SC (`73abfdf`, the head's arg max as the draft):

| Arm | Temperature 0 | Temperature 1.0 |
|---|---|---|
| No drafts | 38.4, 40.1 | 37.0, 37.5 |
| 2 drafts | 52.2, 53.0 (68% kept) | 42.7, 40.7 (50% to 55% kept) |

Gates GA and GB (`b911cb5`), decode in tokens per second and the share of drafts kept:

| Arm | Temperature 0 | Kept | Temperature 1.0 | Kept |
|---|---|---|---|---|
| No drafts | 40.3, 41.4 | | 38.6, 38.1 | |
| 2 drafts | 51.5, 53.1 | 65% | 43.0, 45.8 | 62% to 63% |
| 2 drafts, limit 0.3 | 53.9, 54.1 | 71% | 42.6, 45.5 | 74% to 79% |
| 2 drafts, limit 0.5 | 53.8 | 81% | 41.7 | 80% |
| 2 drafts, limit 0.7 | 51.1 | 90% | 41.8 | 87% |
| 3 drafts | 50.2 | 58% | 43.5 | 57% |

## Notes

- At temperature 0, the text is the same in all 12 runs at `b911cb5`, so the drafts and the gate change no tokens. The
  text differs from the one at `73abfdf`, because the merged projections change the order of one sum.
- Gate GA passes. The limit 0.3 is best at temperature 0 (+32%) and within the noise of the best at 1.0 (+15%).
- Gate GB fails on the Mac: +16% at temperature 1.0, against a bar of 30%. A verify forward of 3 rows costs about as
  much as on the RTX 4080, and sampling keeps fewer drafts than greedy decode.
- 3 drafts are slower than 2 at both temperatures.
- The confidence gate raises the share kept, but it offers fewer drafts, so the speed stays about the same.
- The start script of v0.1.1 sets `Q4E_DRAFTS=2` and `Q4E_DRAFT_MIN_P=0.3` in `run-PACK.sh` on a Mac.
