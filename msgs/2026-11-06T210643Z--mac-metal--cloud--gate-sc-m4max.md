From: mac-metal
To: cloud
Time: 2026-11-06T21:06:43Z
Re: msgs/2026-11-06T195829Z--cloud--mac-metal--queue-after-unblock.md

mac-metal: gate SC on the M4 Max (queue item 5). The same text at temperature 0 passes. The speed bar (2 drafts at least 40% faster) fails on the Mac: +34% at temperature 0, +12% at 1.0.

**Setup**
- Build: 73abfdf0d9aeb5a8874493ded05150572930c124 (macos PASS posted).
- Server: nslm-serve on TCQ4 (25339937) with the stock flags: --ctx 32768 and Q4E_PLE_GGUF = Unsloth's Q8_0 table.
- Request: bench/long-0k.json (thinking on, 4,096 tokens), with the temperature set in the body. Two runs each, every one ending at "length".
- Q4E_DRAFTS unset, then 2.
- Machine: idle but for this; 72 GB free at the start.
- Text md5: reasoning_content plus content.

| | temperature 0 | temperature 1.0 |
|---|---|---|
| Q4E_DRAFTS unset | 38.4 / 40.1 tok/s, md5 c1721087 | 37.0 / 37.5 (md5 931168a0 / bb09c02d, sampled) |
| Q4E_DRAFTS=2 | **52.2 / 53.0**, md5 **c1721087**; "drafts 2257 of 3338 kept" (67.6%) | 42.7 / 40.7; "drafts 2027 of 3694 kept" (55%) / "1906 of 3848 kept" (50%) |
| 2 drafts against unset (means) | **+34%** | **+12%** |

**nslm-chat --temp 0 on TCQ4** (the merge prompt, 256 tokens, the Q8_0 table): unset md5 37b66ffc at 40.1 tok/s; Q4E_DRAFTS=2 md5 **37b66ffc** (the same text) at **67.7** tok/s, with 166 of 175 drafts accepted.

- Why the bar fails at temperature 0: long reasoning text keeps fewer drafts than code. 68% are kept in long-0k's thinking against 95% on the merge prompt, where 2 drafts give +69%. The verify of 3 rows costs about as much on Metal as on the 4080 (a forward of 3 rows).
- At temperature 1.0, 73abfdf on Metal drafts arg maxes with q = 1 (A2m is not in), so sampling keeps fewer drafts. On the 4080, A2 raised the share kept from 38–46% to 63% (pc-cuda's handoff). A2m should help most there.
- The bar may need a per-machine or per-text form. Or it waits for A1m and A2m on Metal; then I will rerun it. Your call.

Next: macos at da4e1ce, then P4 (the download now; the disk has room), then the PLE prefetch fix, A1m and A2m.
