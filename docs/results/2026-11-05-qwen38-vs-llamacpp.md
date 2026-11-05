# Qwen3.8-Flash-Next P3 against llama.cpp UD-Q3_K_XL on one RTX PRO 6000 (2026-11-05)

**Result.** With the routed experts in host memory, the P3 pack fills a 29,120-token prompt at 4,222 tokens per
second. It decodes at 53.8. llama.cpp with Unsloth's UD-Q3_K_XL (`-ot exps=CPU`) does 793 and 9.9 on the same GPU. With
everything in VRAM, llama.cpp is faster: 5,130 and 114 against 4,418 and 87.6. The two packs have about the same
quality: KL divergence against BF16 is 0.0600 for P3 and 0.0593 for UD-Q3_K_XL.

**Status.** Current. One GPU type. The RTX 4080, where Matt sees Strata IQ3_S at about 2,700 and 60 tokens per second,
is not measured here.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell Server Edition, 96 GB). The llama.cpp node had 192 CPU
  threads and 2 TB of RAM. Each engine ran on a different node, at a load average of 1 to 30 (other jobs).
- NanoSeedLM: `30c906c` (the code of `957a0f0`),
  [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) at `5753a67`.
  Host-memory runs from [chunks of 16,384 rows](2026-11-05-qwen38-rows-by-vram.md): an 8 GB expert cache. VRAM runs:
  `--vram-experts` (`scratchpad/rowsjob/vram.sh`). The prompt is the first 120,000 characters of the wikitext-2 test
  set. Decode: 164 tokens of the tide-pools prompt, greedy, with CUDA graphs.
- llama.cpp: `5ce06fc` (CUDA), [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
  at `d962012`, folder `UD-Q3_K_XL` (83.3 GiB). `llama-bench` makes its own prompt tokens. Its decode test (`tg128`)
  starts at an empty context. Script: `scratchpad/rowsjob/llamabench.sh`.

```bash
llama-bench -m UD-Q3_K_XL-00001-of-00004.gguf -ngl 999 -p 29120 -n 0 -b 4096 -ub 4096 -r 2 [-ot exps=CPU]
llama-bench -m UD-Q3_K_XL-00001-of-00004.gguf -ngl 999 -p 0 -n 128 -r 2 [-ot exps=CPU]
```

## Results

Tokens per second:

| Experts | Engine | Prefill, 29,120 tokens | Decode |
|---|---|---|---|
| In host memory | NanoSeedLM P3, an 8 GB expert cache | 4,222 | 53.8 |
| In host memory | llama.cpp UD-Q3_K_XL, `-ot exps=CPU` | 793 | 9.9 |
| In VRAM | NanoSeedLM P3, `--vram-experts` | 4,418 | 87.6 |
| In VRAM | llama.cpp UD-Q3_K_XL | 5,130 | 114.1 |

Quality, from [the packs against BF16](2026-11-04-qwen38-kld-vs-bf16.md) (llama.cpp, 64 windows of 2,048 tokens):

| Pack | Expert bits per weight | Mean KL against BF16 |
|---|---|---|
| UD-Q3_K_XL | 3.69 | 0.0593 |
| P3 | 4.0 | 0.0600 |

## Notes

- With `-ot exps=CPU`, llama.cpp computes the experts on the CPU, here 192 threads. NanoSeedLM copies them to a VRAM
  cache instead: each layer's experts once a prompt pass, and the misses in decode. A desktop CPU would make the
  llama.cpp row slower. A smaller GPU would make both rows slower.
- In VRAM, P3 reaches 86% of llama.cpp's prefill and 77% of its decode. Most of P3's prefill time is in the expert
  and dense GEMMs, and most of its decode time in the matvecs.
- The P3 pack keeps the PLE table in BF16, in host memory. The UD-Q3_K_XL file includes its own PLE table.
- llama.cpp's prompt tokens differ from ours. The two decode tests start at different context lengths (0 and 19
  tokens).
