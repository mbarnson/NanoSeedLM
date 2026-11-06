# Plan: MTP drafts that the sampler keeps more often (2026-11-06)

**Goal.** At the card's sampling defaults (temperature 1.0, `top_p` 0.95, `top_k` 20), `nslm-serve` keeps more of the
MTP head's drafts, as Strata does. Then drafts make decode faster on the RTX 4080 and the M4 Max. Each token keeps the
sampler's distribution. Matt's bar for the RTX 4080 is more than 60 tokens per second for one request. This plan is one
half of that bar. Cheaper verify rows are the other half (see "Verify rows on the RTX 4080").

**Status.** In progress. A1 and A2 are on CUDA (`304d415`, `da4e1ce`), for v0.1.1. On the RTX 4080, GA passes and GB fails: drafts keep 63% to 84%, and decode is break-even ([the gates](../results/2026-11-06-draft-gates-4080.md)). A1m and A2m are for v0.1.2.

## Why

- Gate SC of the serve-drafts plan fails on the RTX 4080 (TCQ3-D4, `bench/long-0k.json`, two runs each):
  - At temperature 0, 2 drafts give the same text and are 3.2% slower.
  - At temperature 1.0, 2 drafts keep 38% to 46% of the drafts, and decode is 16% slower.
  - At temperature 1.0, 1 draft keeps 55% to 58%, and decode is 2% slower.
- The rule keeps a fixed draft d, the head's arg max, with probability p(d). This is speculative sampling with a draft
  distribution q of one point.
- Strata kept 82% of its drafts on the same RTX 4080 ([the probe](../results/2026-11-05-strata-4080-probe.md)). Those
  requests were greedy (temperature 0). Strata's defaults are the card's, as ours are. At temperature 0 the M4 Max kept
  165 of 176 of our drafts. So the gap is at temperature 1.0. Strata's code does two things that ours does not do:
  - It samples the draft from the head's distribution q, and keeps it with probability min(1, p(d) / q(d)).
  - It offers a draft only while the head's probability of it is 0.5 or more (`--spec-min-p 0.5`).

## How it works

Speculative sampling with any draft distribution q (Leviathan et al., 2023):

1. The head samples the draft d from q.
2. The verify row gives the target distribution p: the sampler's temperature and filters on the row's logits.
3. The sampler keeps d with probability min(1, p(d) / q(d)).
4. Otherwise it samples from max(0, p - q), renormalized, and that sample takes the place of d.

Each token then has the distribution p, for any q. A draft is kept with probability 1 - TV(p, q), the sum over tokens of
min(p, q). That sum is larger than p(d) when q is near p. When q is one point at d, the rule is the rule of today. So q
does not have to match the request's filters. Only acceptance depends on q.

The confidence gate does not depend on q. A forward verifies its drafts up to the first one whose head probability is
under a limit. So the forward does not verify the drafts that it would likely reject. The gate does not change the
text: it sets only the number of drafts that a forward verifies.

## Steps and gates

| Step | Work | Who | Gate |
|---|---|---|---|
| A1 | The CUDA arg max kernel of the head also writes the softmax probability of the draft. `q4e_draft_from` returns it. `eng_generate_pick` verifies the drafts up to the first one under `Q4E_DRAFT_MIN_P` | cloud | GA |
| A1m | The same on Metal | mac-metal | GA |
| A2 | Sampled drafts. The GPU samples each draft from q, the head's top 20 at the request's temperature, with a uniform draw from the host. q's 20 ids and probabilities return with the draft. `harness/sample.c` keeps d with min(1, p / q), else samples max(0, p - q) | cloud | GB |
| A2m | The same on Metal | mac-metal | GB |

- **GA.** At temperature 0, the tokens equal the tokens without drafts (`test_qwen4_pick`). On the RTX 4080, measure
  `nslm-serve` with 2 drafts at the limits 0, 0.3, 0.5 and 0.7, at temperature 0 and 1.0. The best limit becomes the
  default. With it, 2 drafts are at least as fast as no drafts, within 2%.
- **GB.** A unit test: for fixed logits, any q and drafts drawn from q, the token frequencies equal p (`test_sample`, as
  gate SB). At temperature 1.0 with 2 drafts:
  - The RTX 4080 keeps at least 65% of the drafts, and decode is at least 5% faster than without drafts.
  - The M4 Max decodes at least 30% faster than without drafts.

## Hypotheses

- **H1.** The head's arg max equals the model's arg max for most tokens: the M4 Max kept 165 of 176 greedy drafts. So q
  is near p, and 1 - TV(p, q) at temperature 1.0 may be 0.7 to 0.85. This is an estimate, not a measurement.
- **H2.** On the RTX 4080, each verify row costs PCIe copies of experts. So the gate may also make drafts faster at
  temperature 0.

## Verify rows on the RTX 4080

Acceptance alone does not reach 60 tokens per second on the RTX 4080. In the probe, a NanoSeedLM step of T rows took
about 9 ms + 25 ms x T. A Strata step took about 10.5 ms + 7.1 ms x T. Each of our rows copies its missed experts over
PCIe. Strata's CPU computes most of its misses while the GPU works. Our CPU tail failed on this PC, because the CPU seed
kernel took 240 to 340 us an expert ([CPU tail](../results/2026-11-05-qwen38-cpu-tail-4080.md)). The release packs use
trellis experts, and the CPU tail does not compute them yet. A CPU trellis kernel, with the misses split between the
CPU and PCIe, needs its own plan. It also makes each row of a batch cheaper.

## Costs

- No HF jobs. Linux CI for each code commit (about $0.08). Runs on the PC and the Mac, about 30 minutes for each step.
- A2 adds one kernel on each GPU: the top 20 of the head's logits, and a sample from them.
- A2 changes the `EngPick` callback: the draft comes with q's list. The greedy pick does not read it.
- Prompt-lookup drafts are not in this plan. Strata uses them too, but they help edits and quotes, not new text.
