# Qwen3.8-Flash-Next on CUDA: cp.async attention, merged GEMMs and a register router (2026-11-05)

**Result.** On an RTX PRO 6000, a 15,922-token prompt of the P3 pack went from 2,801 to 3,403 tokens per second. A
29,120-token prompt went from 2,782 to 3,423, and from 1,571 to 1,691 on an A100. Decode went from 44.0 to 54.2 tokens
per second. Weights of the same input now run as one GEMM, and the router keeps its logits in registers.

**Status.** Replaced by [one-pass prompt attention](2026-11-05-qwen38-one-pass-attention.md).

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell Server Edition, 96 GB), and `a100-large` (A100 80 GB) for
  one row. Each row is a different job and node. The load average was 2 to 8 on the RTX PRO 6000 nodes and 10 to 15
  on the A100 node. Each prompt speed is the mean of two runs, and the two runs agree within 2%.
- Pack: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) at
  `5753a67`, an 8 GB expert cache (3,255 slots).
- Prompts: the first 64,000 characters of the wikitext-2 test set (15,922 tokens, `--ctx 16384`), and the first
  120,000 characters (29,120 tokens, `--ctx 32768`). Decode: 256 tokens of "Why do tide pools matter? Answer in three
  sentences.", with CUDA graphs.
- Scripts: `scratchpad/rowsjob/prof.sh` and `prof32.sh`, as in
  [prefill with larger chunks](2026-11-05-qwen38-prefill-chunks.md).

## Results

The 15,922-token prompt and decode on the RTX PRO 6000:

| Commit | Change | Prompt, tokens/s | Decode, tokens/s |
|---|---|---|---|
| `d8e8a63` | Start ([prefill with larger chunks](2026-11-05-qwen38-prefill-chunks.md)) | 2,801 | 44.0 |
| `6314b52` | Attention: Q fragments in registers, K and V tiles by cp.async (3 blocks an SM) | 2,987 | 44.2 |
| `7908b11` | DeltaNet: the 16-token stages come by cp.async, the next while the block works | 3,038 | 43.8 |
| `aad0df0` | Attention: 4 blocks an SM (128 registers a thread) | 3,140 | 43.8 |
| `80e04f0` | Hyper-connections: the down and inject rows of a mix in one GEMM | 3,269 | 47.6 |
| `0708c6b` | Attention: the row maxima with one TF32 part, not three | 3,296 | 47.3 |
| `7585c47` | One GEMM for each layer's input projections (q, k, v, indexer; qkv, z, b, a) | 3,403 | 49.5 |
| `ff17acd` | Router and shared expert gate in one GEMM, a register router, two fused kernels | 3,377 | 54.2 |

The 29,120-token prompt:

| GPU | `d8e8a63` | `ff17acd` | `470d42a` |
|---|---|---|---|
| RTX PRO 6000 | 2,782 | 3,413 | 3,423 |
| A100 | 1,571 | 1,691 | |

`470d42a` computes the QSA block scores in shared-memory tiles. QSA select at 29,120 tokens went from 370 to 279 ms.

GPU time by stage at `470d42a` (the 29,120-token prompt, RTX PRO 6000):

| Stage | ms | Share |
|---|---|---|
| Experts | 2,855 | 35.1% |
| Projections | 1,427 | 17.5% |
| Attention | 1,362 | 16.7% |
| Hyper-connections | 1,204 | 14.8% |
| Gated DeltaNet | 622 | 7.6% |
| QSA select | 279 | 3.4% |
| Norms, RoPE and stores | 170 | 2.1% |
| PLE | 132 | 1.6% |
| Router and cache | 76 | 0.9% |

## Notes

- An nsys run grouped the dense GEMMs by grid. At 4,096 rows, GEMMs with one block row (32 blocks on 188 SMs) took
  283 ms in 672 calls. Most of it was the HC inject (4 rows, K = 10,240), about 0.64 ms a call. `80e04f0` removes it.
- Decode gains from the merges, because each merge removes launches and a few-row matvec. At `ff17acd` the decode
  stage that holds the router and the shared expert gate fell from 1,145 to 890 ms. The router no longer reads its
  earlier picks from memory in each of its 10 rounds.
- The merges need the weights of one input in one encoding (BF16, Q8 or Q4) and one width. The engine refuses a pack
  where they differ. In P3, the projections are Q4 and the router and shared expert gate are Q8.
- `0708c6b` changed the greedy decode text: 162 tokens became 165. The first token was a near-tie ("are" or "serve
  as"). The kernel tests pass. Pass-0 attention fell 6%, less than its 22% fewer MMAs: the kernel is not MMA-bound.
- Decode at `ff17acd`: router and cache (the expert misses over PCIe) take 37% of the GPU time, experts 24%, and
  hyper-connections 20%. LRU stays the best simple cache policy
  ([cache policies](2026-11-04-qwen38-cache-policies.md)).
- At 29,120 tokens, pc-cuda's GEMM kernels (experts and projections) take 53% of the time.
- An L40S run (Ada, PCIe 4) waited more than 45 minutes for hardware and did not start.
