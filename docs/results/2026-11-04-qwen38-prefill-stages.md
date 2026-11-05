# Long-prompt prefill of Qwen3.8-Flash-Next by stage, and four kernel changes (2026-11-04)

**Result.** On an RTX PRO 6000, a 15,922-token prompt of the P3 pack went from 1,026 to 1,234 tokens per second
(+20%). The attention kernel took most of the gain. One block for each KV group and row now reads each key once for
the group's 12 heads. Before the fixes, the profiler kept only 1,024 events a forward, so it undercounted long prompts
by 62%. Prefill is now GPU-bound: attention 24%, experts 25%, router and cache 18%, hyper-connections 13%.

**Status.** Current. One prompt, one machine type. The RTX 4080 is not measured.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell, 96 GB, PCIe 5), and `a100-large` (A100 80 GB, PCIe 4) for
  two rows. Each row is a different job and node. The load average was 3 to 6.
- Pack: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) at
  `5753a67`, an 8 GB expert cache.
- Prompt: the first 64,000 characters of the wikitext-2 test set (15,922 tokens), two layer-major passes.
- Script: `scratchpad/rowsjob/prof.sh` (`nslm-qwen4-gen` twice with `Q4E_PROFILE=1`, then a 256-token decode).

```bash
Q4E_PROFILE=1 nslm-qwen4-gen --model PACK --prompt "$(cat prompt.txt)" --raw --max-tokens 2 --ctx 16384 --expert-cache-gb 8
```

## Results

Prefill, mean of two runs:

| Commit | Change | RTX PRO 6000, tokens/s | A100, tokens/s |
|---|---|---|---|
| `047ad9f` | main | 1,026 | 539 |
| `ac11534` | PLE rows on 16 threads | 1,083 | 572 |
| `a305bc8` | Attention: a block for each KV group and row | 1,193 | 585 |
| `dd10e71` | DeltaNet: q and k normalised once a token | 1,200 | |
| `89ae78c` | Attention: one softmax step a tile of 16 keys | 1,233 | |
| `d1cff61` | Attention: 16-byte tile loads | 1,234 | |

GPU time by stage at `d1cff61` (RTX PRO 6000, ms for the prompt):

| Stage | ms | Share |
|---|---|---|
| Experts | 3,236 | 25.3% |
| Attention | 3,112 | 24.4% |
| Router and cache | 2,237 | 17.5% |
| Hyper-connections | 1,679 | 13.2% |
| Projections | 1,140 | 8.9% |
| Gated DeltaNet | 1,105 | 8.6% |
| QSA select | 132 | 1.0% |
| Norms, RoPE and stores | 85 | 0.7% |

## Notes

- The PLE rows took 850 ms on one host thread, and 210 ms on 16.
- With 12 query heads for each KV head, each key's K and V now cross L2 once for the group, not once a head. The
  attention stage fell from 4.8 s to 3.1 s.
- The attention kernel uses 72 registers a thread and fits 2 blocks an SM. It runs at about 8% of the SM's issue rate.
  A further gain needs tensor cores over a tile of rows.
- On the A100, the router and cache take 35% (9.2 s): about 118 GB of experts at about 13 GB/s.
- Decode is not slower. It keeps the old attention kernel for up to 8 rows.
