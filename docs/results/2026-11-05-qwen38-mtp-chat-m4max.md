# MTP drafts in nslm-chat on the M4 Max: code 46% faster with 2 drafts (2026-11-05)

**Result.** With `Q4E_DRAFTS=2`, `nslm-chat` decodes code on P3 at 69.6 tokens per second, against 47.8 without
drafts (+46%). The greedy text is the same at every draft count, so gate MB passes in `nslm-chat` too. A short chat
answer is 8.5% faster with 1 draft, 0.5% slower with 2 drafts and 13% slower with 3. Drafts stay off by default.

**Status.** Current.

## Setup

- Machine: mac-metal's Mac. Apple M4 Max (16 CPU cores, 40 GPU cores), 128 GB unified memory, macOS 27.2. No other GPU
  jobs ran during the timed runs. The P3 pack was read into the page cache first.
- Commit: `ba71ba8` (`mac/qwen4-chat-drafts` on main `cb3d6b6`).
- Model: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3), with its
  affine Q4 MTP head.
- Temperature 0. Two rounds. In each round, code then chat, and K = 0, 1, 2 and 3 in that order. K = 0 leaves
  `Q4E_DRAFTS` unset.

```bash
Q4E_DRAFTS=K nslm-chat --model P3 --no-think --max-tokens 256 "Write a Python function that merges two sorted lists, with a docstring."
Q4E_DRAFTS=K nslm-chat --model P3 --no-think --max-tokens 256 "Why do tide pools matter? Answer in three sentences."
```

## Results

Decode tokens per second, as `nslm-chat` reports it (the tokens after the first). The forwards are the forwards after
the first token. The drafts accepted are from round 1, and round 2 gives the same counts. The md5 of stdout is the same
at every K on each prompt.

| Prompt | Drafts | Round 1 | Round 2 | Mean | Forwards | Tokens a forward | Drafts accepted | md5 of stdout |
|---|---|---|---|---|---|---|---|---|
| Code, 256 tokens | 0 | 48.0 | 47.6 | 47.8 | 255 | 1.00 | | `37b66ffc` |
| Code | 1 | 65.0 | 64.8 | 64.9 | 130 | 1.96 | 125 of 130 | `37b66ffc` |
| Code | 2 | 69.4 | 69.7 | 69.6 | 89 | 2.87 | 167 of 176 | `37b66ffc` |
| Code | 3 | 68.7 | 68.6 | 68.7 | 73 | 3.49 | 183 of 203 | `37b66ffc` |
| Chat, 85 tokens | 0 | 47.0 | 47.1 | 47.1 | 84 | 1.00 | | `aafa1411` |
| Chat | 1 | 51.0 | 51.1 | 51.1 | 54 | 1.59 | 32 of 54 | `aafa1411` |
| Chat | 2 | 46.9 | 46.7 | 46.8 | 46 | 1.89 | 42 of 82 | `aafa1411` |
| Chat | 3 | 41.0 | 40.8 | 40.9 | 43 | 2.05 | 46 of 109 | `aafa1411` |

## Notes

- `nslm-chat` takes the first token from `eng_step`, as without drafts, so the time to the first token does not change
  (0.52 s on code). The tokens after it come from `eng_generate`, K + 1 tokens a call.
- `eng_generate` drafts on a pending token when that token is the arg max of the logits that the adapter holds. So a
  call boundary costs no forward without drafts, and the calls can be short. On code at K = 2, calls of 3, 6 and 12
  tokens all took 89 forwards.
- Short calls matter at a stop. The engine computes a call's tokens after the stop, and `nslm-chat` does not show them.
  In these runs, the stop's call held the stop token and at most K more. On chat at K = 2, an earlier build of the same
  day decoded at 45.2, 43.1 and 40.0 tokens per second with calls of 3, 6 and 12 tokens (one round).
- The chat answer accepts fewer drafts than the code answer: 59% at 1 draft, against 96%. So 2 and 3 drafts cost more
  than they save on chat. The share accepted depends on the text, so K is the user's choice.
- The first version of `eng_generate` started each call with a forward of the pending token without drafts, and
  `nslm-chat` called it with 4 (K + 1) tokens. On code, interleaved with `ba71ba8` in three rounds, it gave 47.2, 64.4
  and 67.8 tokens per second at K = 0, 1 and 2, against 47.6, 64.5 and 69.4. `ba71ba8` also halves the time of an arg
  max over the 248,320 logits (0.46 ms to 0.25 ms).
- The prompt template of `nslm-qwen4-gen` differs, so its text and its share of accepted drafts differ too. On its code
  prompt, 2 drafts gave 2.53 tokens a forward ([MTP drafts on the M4 Max, again](2026-11-05-qwen38-mtp-m4max-rerun.md)),
  against 2.87 here.
