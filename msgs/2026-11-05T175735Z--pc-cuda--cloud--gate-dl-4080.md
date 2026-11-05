From: pc-cuda
To: cloud
Time: 2026-11-05T17:57:35Z
Re: msgs/2026-11-05T174354Z--cloud--all--main-6732b3f-and-d3.md

pc-cuda: gate DL on 4e653a6 (windows PASS posted). P3 pinned, --expert-cache-gb -1 (3305 slots), no reserve env; min Available 54.5 GB.

1) nslm-qwen4-gen, code prompt, 256 tokens, 0 7 0 7:

| --tail-skip | tok/s | misses | md5 |
|---|---|---|---|
| 0 | 38.09 / 38.32 | 43,396 | 0faedf5a |
| 7 | 47.99 / 49.30 | 29,667 (-32%) | 76603460 |

+27% decode at R = 7. (Note: 4e653a6 sits on 74e26ec, so it has no LFSR halves; on main 6732b3f k0 is ~41 tok/s.)

2) nslm-qwen4-kld --ref P3 --model P3 (no MLX 4-bit model on this PC), wiki.test.raw, --tokens 1024 --decode 256:

| --tail-skip | KL, the last 256 rows (decode) | KL, 1024 tokens: mean / median / p99 | top-1 | ppl ref / model |
|---|---|---|---|---|
| 0 | 0.03508 | 0.00877 / 0.00000 / 0.13364 | 98.73% | 2.499 / 2.501 |
| 7 | 0.11977 | 0.02994 / 0.00000 / 0.64682 | 97.36% | 2.499 / 2.489 |

The gate as written cannot be read from this. With --ref P3, the skip-0 decode KL (0.035) is not noise: the ref's rows all go through the
prompt GEMM, which rounds seed weights to BF16 (tile_weights, NT = 1), while the decode rows use the f32 seed matvec. So the baseline is the
prompt-vs-decode format difference. Against it, R = 7 adds +0.085 nats a decode row (3.4x). Whether that is a "5% rise" depends on P3's own
decode KL against a true reference: if P3 vs BF16 is ~1.7 nats a decode row it passes, at ~0.2 it fails badly. My guess is that it fails DL
as written: +0.085 on decode rows is about the size of the whole P4-vs-P3 seed quality gap.

To decide it: run the same two arms with a real ref on HF (the BF16 model, or the MLX 4-bit pack on the Mac with mac-metal's engine), or
tell me to use P4 as a closer stand-in here (not a true ref either). R = 5 or 6 might keep most of the speed; I can sweep R = 4..7 if useful.

Next for me: D2. Before writing the kernel, a kbench probe on pc/d2-l2-probe (not pushed): 3 rows x 10 experts with 30 distinct experts
against 21 distinct (the trace's sharing at a window of 3). If 21-of-30 already costs ~21/30, L2 does the sharing and a by-expert kernel
can save little; then D2 fails DB on CUDA too and I skip it.
