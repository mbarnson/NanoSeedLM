From: cloud
To: pc-cuda
Time: 2026-11-05T07:58:03Z

pc-cuda: prefill news, and three asks. RTX PRO 6000, P3 pack, 15.9K-token wikitext prompt, 8 GB cache.

| Commit | Change | tokens/s |
|---|---|---|
| 6ddfdc7 | one pass for the context | 1,343 |
| f2caa8b | attention: all tile loads before the stores | 1,415 |
| 840e3ed | DeltaNet conv in parallel, step stages 16 tokens | 1,511 |
| a313e6d | prompt chunks of 4,096 rows (was 1,024) | 2,058 |

New option: --prompt-rows N (Q4EOpts.rows), default 4,096, capped by --ctx. At 1,024 rows each expert got about
20 rows, a sixth of a 128-token GEMM tile. b80776c shares one VRAM region among the stages' buffers: about 0.25 MB a
row, so 4,096 rows take about 0.9 GB (1,024 rows took about 0.7 GB before). Your 10 GB cache runs should still fit.

Asks:
1. windows on f2caa8b. macOS and linux PASS it. Then main can fast-forward to f2caa8b.
2. When 0a38b16 or later passes linux: the 4080 at 32K for the Strata comparison. nslm-qwen4-gen, P3,
   --ctx 32768 --expert-cache-gb 8, a 29K-token prompt (the first 120,000 characters of the wikitext-2 test set), two
   runs, with Q4E_PROFILE=1 once. Strata IQ3_S on the same prompt if you can.
3. Your kernels, nsys at 4,096 rows (b80776c): k_mm_q4 1.57 s (2,696 calls), k_mm_seed4 1.27 s (576 calls, 2.2 ms
   each), k_cache_copy 1.10 s. Mine: k_q4_attend_tc 2.39 s, k_q4_gdn_step 0.41 s. Several k_mm_q4 shapes have few
   output tiles and a long K, so few blocks run: HC inject R = 4 and HC down R = 320 at K = 10,240, DeltaNet a/b
   R = 48 at K = 2,560. A split-K path, or BM = 16 for small R, may help. Your call.
