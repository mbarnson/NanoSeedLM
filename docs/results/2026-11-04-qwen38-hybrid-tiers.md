# Hybrid P3 / P4 seed packs of Qwen3.8-Flash-Next against BF16 (2026-11-04)

**Result.** These packs keep a third of the experts in P4 and the rest in P3: 4.17 bits for each expert weight. With
a random third, the mean KL divergence is 0.054. It is the same when the third is every `down_proj`, or the experts that
chat and code text use most. When the third is the experts that wiki text uses most, the KLD on wikitext falls to
0.0450. That is 17% less, and below Unsloth's curve at the same bits (0.0465).

**Status.** Current. One text. The in-domain map comes from other articles of the same source.

## Setup

- Machine, llama.cpp, reference, text and emulation: as in
  [the packs against BF16](2026-11-04-qwen38-kld-vs-bf16.md), job `6aebfd40c78863f9c6332197`.
- Tiers: `nslm-qwen4-emulate --pack P3 --pack P4 --map MAP` takes each expert of each projection from the pack that
  the map names. `tools/qwen4_tier_map.py` writes the maps:
  - `down`: every `down_proj` in P4.
  - `random`: 170 random experts of each layer in P4 (all three projections).
  - `freqcc`: the 170 most used experts of each layer on the chat and code traces.
  - `freqwiki`: the 170 most used experts on the wiki trace (wikitext-103 train).
- Traces: [expert use by text domain](2026-11-04-qwen38-routing-domains.md).

```bash
python3 tools/qwen4_tier_map.py --mode freq --trace p3-wiki.bin --out maps/freqwiki.bin
hf jobs run --flavor rtx-pro-6000x4 -v hf://buckets/txgsync/nslm-qwen38:/b -e SRC=/b/src-e8426e0 -e OUT=/b/kld-t5a \
    -e MAPS=/b/maps -e "PACKS=Q8_0 P3 P4 H-down H-random H-freqcc H-freqwiki" \
    nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /b/src-e8426e0/tools/hf_llama_kld.sh
```

## Results

| Pack | Expert GB | Bits for each expert weight | P4 share of wiki router choices | Mean KLD | Median KLD | 99th percentile KLD | Top-1 agreement |
|---|---|---|---|---|---|---|---|
| P3 | 60.40 | 4.000 | 0% | 0.0600 | 0.0159 | 0.723 | 91.55% |
| down | 62.91 | 4.167 | (all `down_proj`) | 0.0533 | 0.0139 | 0.653 | 91.99% |
| random | 62.92 | 4.167 | 33.5% | 0.0540 | 0.0139 | 0.666 | 91.99% |
| freqcc | 62.92 | 4.167 | 39.8% | 0.0540 | 0.0139 | 0.662 | 91.99% |
| freqwiki | 62.92 | 4.167 | 84.0% | 0.0450 | 0.0117 | 0.548 | 92.82% |
| P4 | 67.95 | 4.500 | 100% | 0.0408 | 0.0106 | 0.484 | 93.13% |

The router shares are measured on the wiki trace. The standard error of each mean KLD is 0.0005 to 0.0007.

## Notes

- The KLD follows the share of router choices that P4 serves. `down_proj` gains nothing over a random third.
- A pack for all uses cannot fix the hot set: chat and code choose other experts than wiki text. The expert cache
  follows the hot set of each session, so a tier that follows the cache may give the in-domain result. That needs
  both tiers of each expert and an engine change. It is not measured.
- Unsloth's curve: [the packs against BF16](2026-11-04-qwen38-kld-vs-bf16.md), between UD-IQ4_XS and UD-Q4_K_XL.
