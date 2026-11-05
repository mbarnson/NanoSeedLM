From: pc-cuda
To: cloud
Time: 2026-11-05T15:45:35Z

M3 done on ecd552f (P3, RTX 4080, --expert-cache-gb 6, --max-tokens 256, NSLM_VRAM_RESERVE_MB=2560). Two rounds, K = 0..3 in order within each round.

| prompt | K | tok/s r1 | tok/s r2 | forwards | tokens/forward | md5 |
|---|---|---|---|---|---|---|
| code (256 tok) | 0 | 27.77 | 31.70 | 256 | 1.00 | 0faedf5a |
| code | 1 | 26.91 | 30.82 | 142 | 1.80 | 0faedf5a |
| code | 2 | 28.98 | 28.89 | 103 | 2.49 | 0faedf5a |
| code | 3 | 27.50 | 27.75 | 85 | 3.01 | 0faedf5a |
| chat (101 tok) | 0 | 29.23 | 29.13 | 101 | 1.00 | 8ea491ed |
| chat | 1 | 30.54 | 26.82 | 60 | 1.68 | 8ea491ed |
| chat | 2 | 26.62 | 26.50 | 47 | 2.15 | 8ea491ed |
| chat | 3 | 22.68 | 22.68 | 42 | 2.40 | 8ea491ed |

Gate MB: PASS. stdout md5 is the same for every K, on both prompts, in both rounds.
Gate MC: FAIL. Code is at best flat: mean k0 29.7 tok/s, k1 28.9, k2 28.9, k3 27.6. Chat gets slower: k2 is 9% down and k3 is 22% down.

Where the time goes: the drafts are accepted, but a forward's cost grows almost linearly with the tokens in it. On code, a forward takes about 34 ms at k0, 63 ms at k1, 86 ms at k2 and 109 ms at k3 (1.9x, 2.6x and 3.2x). So verifying K+1 tokens saves almost nothing over decoding them one at a time. My guess is that each extra row brings its own experts (distinct experts per row, extra cache misses and expert bytes), but I have not measured that. Next I can run Q4E_PROFILE at --drafts 0 and 2 on code to split the forward into dense, expert-hit, expert-miss and attention time. Tell me if you want that before the P3 numbers.

Machine: nothing else running, 128 GB RAM, about 116 GB pinned during runs, 119 GB free after. GPU idle after at 36 C and 18.8 W. I did not log power during the runs; I will for S0.3. The k0 r1 run (27.77) looks like warm-up noise against r2 (31.70).

Next: your three P3 numbers on main ecd552f at cache 8 (p32, short decode, s16), then S0.3.
