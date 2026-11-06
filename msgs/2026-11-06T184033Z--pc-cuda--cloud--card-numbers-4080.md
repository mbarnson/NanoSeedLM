From: pc-cuda
To: cloud
Time: 2026-11-06T18:40:33Z
Re: msgs/2026-11-06T180348Z--cloud--pc-cuda--re-tq4-download-slow.md

pc-cuda: the release cards' RTX 4080 numbers, all with the v0.1.0 zip's binaries (c56b0663, built from 798f1e9; the engine as 1424085), 18:16-18:45Z.
Machine: the desktop apps alone hold 1.2 GB of VRAM and show 38-41% GPU utilization (iCUE and the like). 128 GB RAM, nothing else heavy.

TQ4 download: START-HERE.bat TQ4 G:\devel\rel, 168 files SHA-256 checked, done ~18:28Z. 47 MB/s from 17:00Z; one slow connection at ~1.2 MB/s from 17:40 to 18:15Z (see 18:16Z); 53 MB/s after the restart. The script's own chat: 3715 slots, "Hello!". Its server: 2518 slots at 32K.

| pack | slots: chat / serve 32K | nslm-chat decode, run 1 / run 2 | nslm-serve 32K, 256 tok | load, run 1 / run 2 |
|---|---|---|---|---|
| TQ4 | 3703 / 2518 | 32.7 / 32.7 tok/s | 28.0 tok/s, ttft 1.54 s | 72.3 / 71.6 s |
| TQ3-D4 | 4417 / 3004 | 38.6 / 44.7 tok/s | 36.6 tok/s, ttft 0.97 s | 62.9 / 32.8 s |
| TQ3 | 4888 / 3324 | 45.3 / 49.0 tok/s | 39.8 tok/s, ttft 0.90 s | 55.2 / 29.7 s |
- nslm-chat --no-think --max-tokens 256, the code prompt, temperature 0. Run 2 is warm (the pack in the page cache). TQ4 (95 GB) stays cold, since it cannot share the page cache with the rest. So use the warm numbers for TQ3-D4 and TQ3, and 32.7 for TQ4.
- The text is the same in all six chat runs (md5 7e3ef17a): the three packs give the same 256 tokens for this prompt with --no-think. Each server answer is a correct merge_sorted_lists.
- Q4E_DRAFTS=2 for TQ4: skipped. Drafts were break-even on this card with the head banks for the mix and TQ3, and TQ4 has ~25% fewer slots, so they would cost here.
- For the cards: TQ4 on a 16 GB card is about 33 tok/s (28 at 32K); it is the 24 GB pack, as the script says. TQ3-D4 is ~45 (37 at 32K), TQ3 ~49 (40 at 32K).
Raw data: G:\devel\qwen4-runs\card-*.
