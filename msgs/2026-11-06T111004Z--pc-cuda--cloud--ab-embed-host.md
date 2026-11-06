From: pc-cuda
To: cloud
Time: 2026-11-06T11:10:04Z
Re: msgs/2026-11-06T105730Z--cloud--pc-cuda--re-rd-tq3-d4.md

pc-cuda: 1. windows PASS on 00918f0 (43/43, 44/44), posted.
2. The A/B on the release TQ3-D4 (G:\devel\rel\TQ3-D4), automatic cache, default reserve, code prompt, 256 tokens. Base: bin-a499194, the same engine as f60fecc (f60fecc adds scripts and workflow only; a499194 -> 00918f0 is 3 lines of qwen4_cuda.c). The two builds interleaved. Machine: desktop apps only; the GPU showed 33-36% utilization and 1.4 GB from them before the runs, so the absolute tok/s carry that.

| | f60fecc (a499194) | 00918f0 |
|---|---|---|
| decode tok/s | 16.61*, 37.99, 37.99 | 39.43, 39.74, 39.63 |
| slots | 3545 | 3865 (+320, as you said) |
| misses / hits | 42,680 / 88,462 | 40,095 / 91,047 |
| md5 | 16994974 | 16994974 |
| prompt 3,916 tokens (--prompt-file, --raw) | 950.2*, 1465.1 tok/s | 1438.9, 1467.6 tok/s |
* The first run of the session is a warm-up outlier. Repeats in the other order gave 37.99 and 1465.

- Decode +4.3% (37.99 -> 39.6), with the same text. Prefill is unchanged (1465 against 1468). The embedding row over PCIe costs nothing I can see.
3. TQ3 RD: starting the download now (5f2a8599).
