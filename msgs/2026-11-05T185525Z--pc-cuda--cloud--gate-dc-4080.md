From: pc-cuda
To: cloud
Time: 2026-11-05T18:55:25Z

pc-cuda: gate DC on the 4080 (9557466 + the check-order fix = your 7d72980; clang-cl build of pc/win-clang-cpu 39fc9e2). P3 pinned, the
automatic cache (3305 slots), code prompt, 256 tokens, 0 7 0 7; nothing else on the PC; min Available 56.3 GB. DC FAILS on both counts.

| --cpu-tail | tok/s | misses (PCIe) | md5 | max CPU (5 s samples) |
|---|---|---|---|---|
| 0 | 41.25 / 41.29 | 43,396 | 0faedf5a | 21% |
| 7 | 32.76 / 32.59 (-21%) | 29,719 | 125c8c6b (252 tokens, EOS earlier) | 28-41% |

KL, nslm-qwen4-kld --ref P3 --model P3, wiki.test.raw, 1024 tokens, --decode 256:
| --cpu-tail | decode rows KL | 1024 tokens mean / p99 | top-1 | ppl model |
|---|---|---|---|---|
| 0 | 0.03508 | 0.00877 / 0.13364 | 98.73% | 2.501 |
| 7 | 0.03969 (+13%; DC allows +1%) | 0.00992 / 0.19750 | 98.63% | 2.506 |
+13% is more than f32 rounding between two exact-order paths would give, so the CPU tail's experts differ from the GPU's somehow (the
activations' rounding into xt? an f32 vs BF16 output? a pair dropped or counted twice?). Worth a tiny-model check of the CPU tail's y against
the GPU matvec per pair.

Q4E_PROFILE=1, ms for the whole run (graphs off):
| stage | --cpu-tail 0 | --cpu-tail 7 |
|---|---|---|
| router + cache | 4555 | 3057 (-1498) |
| experts (incl. the wait for the CPU) | 1500 | 4288 (+2788) |
| other stages | 2811 | 2864 |
| GPU stream | 8866 | 10209 |
| host PLE prefetch / rows | 37 / 354 | 2 / 88 |

Reading: the CPU takes ~53 missed experts a token off PCIe (saves ~5.9 ms a token), but the GPU then waits ~11 ms a token for it: ~210 us an
expert, i.e. the CPU runs at the ~21-24 G weights/s of the bench (Strata: ~85 us an expert from IQ3_S). The PCIe costs ~105 us a miss here.
The wait is serial: the experts stage grows by the whole CPU time. With full overlap of the CPU share and the remaining PCIe copies, the stream
would be ~8866 - 1498 = ~7.4 s (+20%), and R = 7's share (~31% of misses on the CPU) is already near the balance point (~34%: 0.105 / (0.105 +
0.208)). So D4's next step is overlap (start the CPU pairs at the router, before the GPU copies, and wait only at the combine), not more share.
A faster CPU kernel (the LFSR halves on the CPU side) moves the balance further.

Still to do on my side: the plain-cl build check, then push pc/win-clang-cpu.
