# Plan: MTP drafts in nslm-serve, with sampling (2026-11-06)

**Goal.** `nslm-serve` decodes Qwen3.8-Flash-Next with the MTP head's drafts, at any temperature, with the output
distribution of its sampler without drafts. On the M4 Max, `nslm-chat` decodes TCQ4 at about 40 tokens per second, and
at 67 to 69 with 2 drafts. The server is what most users run.

**Status.** Draft. For v0.1.1, after v0.1.0.

## Why the server does not draft

- `eng_generate` keeps a draft while it equals the arg max of its verify row. That is exact at temperature 0 only.
- `nslm-serve` samples at the card's defaults (temperature 1.0, `top_p` 0.95). It decodes with `eng_step`, one token at
  a time, so the drafts were left out instead of made exact for sampling.
- `q4e_draft` takes its first input, the bonus token, from an arg max on the GPU. A sampled token is often not that.

## How it works

Speculative sampling with a single draft token (Leviathan et al., 2023). The verify forward gives the logits of each
row. For row i, the sampler makes the target distribution p (temperature, `top_k`, `top_p`, `min_p`). It keeps the
draft d with probability p(d). Otherwise it samples from p without d, renormalized, and the row's token is that sample.
When all drafts hold, the next token is a sample from the last row. Each token then has the distribution of the sampler
without drafts. At temperature 0, p is one-hot, so the rule is today's greedy rule.

## Steps and gates

| Step | Work | Who | Gate |
|---|---|---|---|
| S1 | `q4e_draft(e, bonus, k, drafts)`: the caller gives the first input token (CUDA: a copy in place of the arg max) | cloud | SA |
| S1m | The same in `qwen4_gpu.m` | mac-metal | SA |
| S2 | `eng_generate_pick`: `eng_generate`'s loop with a pick callback. `eng_generate` passes the arg max | cloud | SA |
| S3 | `harness/sample.c`: one sampler (`nslm-serve`'s filters) and its draft rule, for `nslm-chat` and `nslm-serve` | cloud | SB |
| S4 | `nslm-serve`: with `Q4E_DRAFTS`, a Qwen slot decodes with `eng_generate_pick` | cloud | SC |

- **SA.** At temperature 0, the drafted tokens equal the tokens without drafts (gate MB of the MTP plan), on CUDA and
  Metal.
- **SB.** A unit test with given uniform draws u. For fixed logits and a draft d, the rule keeps d exactly when
  u < p(d). Otherwise it samples p without d. At temperature 0 it gives the arg max.
- **SC.** On the M4 Max, `nslm-serve` with 2 drafts decodes TCQ4 at least 40% faster than without. That holds at
  temperature 0, with the same text, and at temperature 1.0. On the RTX 4080 it is at most 3% slower. If not, drafts
  stay off there by default.

## Costs

- No HF jobs. Linux CI for each code commit (about $0.08), and runs on the PC and the Mac.
- `nslm-serve` with drafts keeps one sequence slot for Qwen, as now.
