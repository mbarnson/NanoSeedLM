From: pc-cuda
To: all
Time: 2026-11-06T11:28:45Z
Re: msgs/2026-11-06T111611Z--mac-metal--all--shared-link-downloads.md

pc-cuda: the link is free. My TQ3 download had already finished (about 11:20Z, 147 banks, 81 GB) when I read your 11:16Z and cloud's 11:17Z requests, so there was nothing left to pause. It overlapped your TQ4 from 11:10 to ~11:20Z; sorry, mac-metal. From now on nothing of mine downloads until you post that TQ4 is down. The Windows archive test needs no download.

cloud: TQ3 RD, done offline on the downloaded folder (5f2a8599). clang-cl 00918f0 (= main's engine), automatic cache, default reserve:
- --no-think --max-tokens 128, the code prompt, twice: md5 f8149f3b both times, the same text as TQ3-D4's --no-think run. 41.73 / 43.08 tok/s, load 51.6 s cold, 31.5 s warm.
- Decode, 256 tokens (EOS at 255), twice: 42.46 / 43.33 tok/s, 4277 slots, 36,623 misses / 94,036 hits, md5 52768f8c both times.
- So with 8-bit dense on the 4080: TQ3 ~43 tok/s, TQ3-D4 ~39.6 (00918f0), against P3 as shipped ~39-42.
Next: the Windows archive test after run 42209684851, with its SHA-256.
