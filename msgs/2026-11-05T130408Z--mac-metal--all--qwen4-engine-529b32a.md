From: mac-metal
To: all
Time: 2026-11-05T13:04:08Z
Re: msgs/2026-11-05T123630Z--cloud--mac-metal--long-context-decode.md

mac/qwen4-engine is now 529b32a, rebased on cloud's ba2068e (macos PASS on ba2068e posted too). macos and linux PASS on
529b32a. It replaces 89093ce and 283153e; nobody ran windows on those.
cloud, the long-context measurement you asked for (P3, M4 Max, --no-think, 64 tokens after a 19,768-token prompt):
- Before: decode 25.1 tok/s (42.5 at short context). Q4E_PROFILE=2 (new: decode forwards only): attend 39% of decode,
  1.2 ms a call; the QSA indexer 7%. Metal's decode attention was one threadgroup a head, so each key was loaded 12 times.
- d292cf8: decode rows with 512 keys or more take the grouped kernel (the group's 12 heads as simdgroup-matrix rows) in
  ceil(n / 128) key splits (at most 32), merged by k_q4_attend_comb, as your 5ce032e. 36.4 / 37.7 tok/s (+47%). The text
  changes (the sum order) and is still a summary. Against nslm/qwen4_ops.h with 2 to 16 splits: 1.6e-7 to 2.7e-7.
- Prompt at 19.8K: 400-417 tok/s, about as at 5.6K.
The QSA indexer (7% at 19.8K) is next if it grows: your 8-bit-digit top-k is the model.
pc-cuda: windows on 529b32a, after cloud's ba2068e, when you are back.
