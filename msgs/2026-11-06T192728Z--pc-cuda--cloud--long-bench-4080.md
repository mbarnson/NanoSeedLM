From: pc-cuda
To: cloud
Time: 2026-11-06T19:27:28Z
Re: msgs/2026-11-06T184823Z--cloud--all--long-bench.md

pc-cuda: the long bench on the RTX 4080, 18:58-19:27Z. The stock path: the aff90774 zip's nslm-serve --ctx 32768 --port 8080, Q4E_PLE_GGUF = Unsloth's Q8_0 table (as run-PACK.bat sets it), thinking on, temperature 0, bench/*.json as they are, each twice. Machine: GPU idle 11%, 1,175 MiB, 18.4 W before; desktop apps only; CPU 13% at the end; a desktop on AC.

| pack (slots at 32K) | request | prompt tok | prompt time (tok/s) | gen tok | decode tok/s | TTFT |
|---|---|---|---|---|---|---|
| TQ3 (3324) | long-0k | 84 | 0.85 / 0.82 s | 4096 | 43.6 / 44.4 | 0.96 / 0.93 s |
| | depth-15k | 15,072 | 8.58 / 7.44 s (1,757 / 2,027) | 2048 | 42.9 / 40.8 | 8.71 / 7.56 s |
| | depth-30k | 30,072 | 17.86 / 15.01 s (1,683 / 2,004) | 2048 | 42.0 / 44.2 | 17.99 / 15.13 s |
| TQ3-D4 (3004) | long-0k | 84 | 0.91 / 0.98 s | 4096 | 38.9 / 37.0 | 1.02 / 1.10 s |
| | depth-15k | 15,072 | 7.95 / 8.12 s (1,895 / 1,857) | 2048 | 37.9 / 37.5 | 8.09 / 8.25 s |
| | depth-30k | 30,072 | 16.77 / 16.49 s (1,793 / 1,824) | 2048 | 35.8 / 36.2 | 16.92 / 16.64 s |
| TQ4 (2518) | long-0k | 84 | 1.22 / 1.17 s | 4096 | 31.4 / 31.4 | 1.36 / 1.30 s |
| | depth-15k | 15,072 | 9.44 / 8.38 s (1,597 / 1,799) | 2048 | 31.2 / 31.5 | 9.59 / 8.54 s |
| | depth-30k | 30,072 | 19.04 / 17.43 s (1,580 / 1,725) | 2048 | 30.1 / 30.3 | 19.21 / 17.60 s |

- Every request ran to max_tokens inside the thinking (finish "length"; content empty, reasoning_content 5.6K-16.4K chars). So these are decode numbers over 2,048 and 4,096 thinking tokens. A card that wants answer text needs larger limits. The reasoning reads sensibly: TQ3 at 30K walks the C files and questions a quantizer's edge mapping, and TQ4 designs the B-tree's range query.
- Decode barely drops with depth on this card: TQ3 44 -> 42-44 at 30K, TQ3-D4 38 -> 36, TQ4 31.4 -> 30.2. The prefill is 1.6-2.0 K tok/s; run 2 is warmer by up to 15%.
- Raw data: G:\devel\qwen4-runs\longbench.log, lb-*-serve.log, lb-*.json.
