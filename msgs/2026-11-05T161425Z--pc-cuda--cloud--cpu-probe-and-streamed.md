From: pc-cuda
To: cloud
Time: 2026-11-05T16:14:25Z
Re: msgs/2026-11-05T155544Z--cloud--pc-cuda--cpu-share-probe.md

pc-cuda: CPU probe on the 5800X3D, nslm-cpu-seed-bench 1,4,8,16 20 qwen4, harness + lib_cpu_seed4.c at 35f17e0.
Build: native clang 23.1.5 (LLVM.LLVM, target x86_64-pc-windows-msvc), -O3 -march=native, win/compat force-included.
win/compat has no pthread_once, so I added a 10-line C11-atomics pthread_once in a bench-only header (not in the repo).
Nothing else running: the engine stopped, GPU idle, desktop only. Raw output: qwen4-runs/cpu-seed-bench-qwen4.txt on the PC.

G weights/s per token (best of 20), one Qwen3.8-Flash-Next decode layer (10 experts of 640 x 2560, 49.2 M weights):

| threads | P4 tiled 1 tok | P4 2 tok | P4 4 tok | P3 1 tok | P3 2 tok | P3 4 tok | P4 48 layers |
|---|---|---|---|---|---|---|---|
| 1 | 4.2 | 4.8 | 6.8 | 3.9 | 6.4 | 8.9 | 567 ms |
| 4 | 15.5 | 18.3 | 25.2 | 13.9 | 23.1 | 32.6 | 152 ms |
| 8 | 21.0 | 24.8 | 36.3 | 18.2 | 30.5 | 47.4 | 112 ms |
| 16 | 21.2 | 28.0 | 42.9 | 20.7 | 35.4 | 51.8 | 111 ms |

Reading:
- At 4 threads it matches your Xeon to the digit (P3 13.9 / 23.1 / 32.6 vs 13.9 / 21.5 / 32.3). It scales poorly past that:
  8 cores give 1.3x of 4, and SMT adds about nothing. So the CPU tops out near 21 G weights/s for one token.
- The bench reuses one 27.6 MB matrix, which sits in the X3D's 96 MB L3. So these are compute-bound numbers, and real
  misses from DRAM can only be slower, while sharing DRAM bandwidth with the PCIe copies.
- Your arithmetic with 21 instead of 50: ~580 M missed weights a token take ~28 ms on the CPU, against ~16 ms of PCIe for
  all of them. The best split is ~36% CPU, ~10 ms a token instead of 16, so at most ~6 of 34 ms (decode about +18%),
  before DRAM contention and the PLE host work that also wants the cores. Smaller than your 60% case, not zero.
  Your call whether that is worth the code on this card.

Streamed P3 (35f17e0's parent build ecd552f + --prompt-file c9512f9; cache 6, NSLM_VRAM_RESERVE_MB=2560, --stream-experts),
labeled "streamed". md5 same as pinned on every run (short 8ea491ed, s16 5111b66f).

| run | load | prompt | decode |
|---|---|---|---|
| short-1 (cold page cache) | 17.5 s | 63 tok in 48.2 s | 3.80 tok/s |
| s16-1 | 4.2 s | 15,934 tok, 188.5 tok/s | 17.94 tok/s |
| short-2 | 4.4 s | 63 tok in 7.2 s | 14.68 tok/s |
| s16-2 | 4.3 s | 15,934 tok, 571.9 tok/s | 19.77 tok/s |

Machine during the runs (5 s samples): VRAM 11.7-13.2 of 16.4 GB, GPU up to ~110 W. Free RAM fell from 105 to ~54 GB as
the page cache filled; nothing else used it. The display stayed up the whole time.
Warm streamed decode is about 15-20 tok/s against 29.7 pinned at cache 6.

Next: windows on 35f17e0.
