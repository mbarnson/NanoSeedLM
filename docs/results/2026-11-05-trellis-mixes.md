# Trellis packs at 2 to 4 bits and Unsloth's curve, corrected: KL in llama.cpp (2026-11-05)

**Result.** Trellis-coded experts lie below Unsloth's curve at every bit rate that the curve covers. At 3.125 bits the
KL is 16% below it, and at 4.125 bits 17% below, about 0.17 and 0.40 bits saved. K = 4 for every `down_proj` and
K = 3 for the rest gives 0.0619 at 3.46 bits, 10% below the curve. Unsloth's expert sizes were too small on earlier
pages: a new GGUF type counted as 0 bytes. At 2.1 and 2.5 bits the KL is 0.204 and 0.164, below any Unsloth pack
that we measured.

**Status.** Current. Steps T2 and T4 of the [trellis plan](../plans/2026-11-05-trellis-experts.md), at L = 12. It
corrects the curve of [Trellis-coded experts against BF16](2026-11-05-trellis-kld.md).

## Setup

- KL: llama.cpp against BF16 with the BF16 PLE table, as in [the packs against BF16](2026-11-04-qwen38-kld-vs-bf16.md).
  `nslm-qwen4-emulate` writes the trellis experts into the Q8_0 expert tensors of Unsloth's Q8_0 shards.
- Job: HF Jobs `rtx-pro-6000x4`, `6aed21aec78863f9c633bdd0` at `7e442c7`, 29 minutes, $5.37.
- Banks: L = 12. K = 2 from job `6aed1ef1c78863f9c633bc09` (8 GPUs, 7.5 minutes). K = 3 and K = 4 from
  [Trellis-coded experts against BF16](2026-11-05-trellis-kld.md).
- Mixes: two banks and a tier map of `tools/qwen4_tier_map.py` (as for the [hybrid tiers](2026-11-04-qwen38-hybrid-tiers.md)).
  - `down`: the larger K for every `down_proj`, the smaller K for `gate_proj` and `up_proj`.
  - `freqcc`: the larger K for the third of each layer's experts that a routing trace of another text uses most.
- Unsloth's expert bits: `tools/gguf_expert_bytes.py`, each expert tensor's bytes from the offsets in the GGUF
  headers, read over HTTP. Type 42,
  which our `gguf` package (0.19.0) does not know, takes 2.25 bits for each weight. The known types give their
  standard sizes (IQ3_XXS 3.0625, IQ4_NL 4.5).

```bash
hf jobs run --flavor rtx-pro-6000x4 -v hf://buckets/txgsync/nslm-qwen38:/b -e SRC=/b/src-7e442c7 -e OUT=/b/kld-tq2 \
    -e TQBANK=/b/tq -e MAPS=/b/maps \
    -e "PACKS=Q8_0 TQ2L12 TQ2L12+TQ3L12@freqcc TQ3L12+TQ4L12@down TQ3L12+TQ4L12@freqcc UD-Q2_K_XL" \
    nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /b/src-7e442c7/tools/hf_llama_kld.sh
```

## Results

KL against BF16. Rows without this job come from the pages above. Q8_0 reproduced its mean KL (0.01986).

| Pack | Expert GB | Bits for each expert weight | Mean KLD | Median KLD | 99th percentile KLD | Top-1 agreement |
|---|---|---|---|---|---|---|
| Q8_0 | 128.35 | 8.500 | 0.0199 | 0.0051 | 0.228 | 95.23% |
| UD-Q4_K_XL | 77.02 | 5.101 | 0.0304 | | 0.362 | 93.97% |
| P4 (seeds) | 68.0 | 4.500 | 0.0408 | | 0.484 | 93.13% |
| **Trellis K = 4** | **62.29** | **4.125** | **0.0398** | | **0.479** | **93.06%** |
| UD-IQ4_XS | 60.38 | 3.999 | 0.0509 | | 0.608 | 92.31% |
| P3 (seeds) | 60.4 | 4.000 | 0.0600 | | 0.723 | 91.55% |
| UD-Q3_K_XL | 56.18 | 3.720 | 0.0593 | | 0.697 | 91.60% |
| **Trellis K = 3 + 4, `down`** | **52.22** | **3.458** | **0.0619** | **0.0169** | **0.755** | **91.44%** |
| **Trellis K = 3 + 4, `freqcc`** | **52.23** | **3.459** | **0.0664** | **0.0175** | **0.811** | **91.23%** |
| UD-IQ3_XXS | 50.29 | 3.331 | 0.0735 | | 0.865 | 90.79% |
| **Trellis K = 3** | **47.19** | **3.125** | **0.0765** | | **0.931** | **90.50%** |
| UD-Q2_K_XL | 46.03 | 3.049 | 0.0994 | 0.0293 | 1.142 | 89.09% |
| **Trellis K = 2 + 3, `freqcc`** | **37.21** | **2.465** | **0.1642** | **0.0475** | **1.921** | **86.28%** |
| **Trellis K = 2** | **32.21** | **2.133** | **0.2040** | **0.0640** | **2.262** | **84.34%** |

The standard error of each mean KLD is 0.0003 to 0.0018.

Unsloth's curve, interpolated linearly in log KL against bits between its two nearest packs (gate TB):

| Pack | Bits | The curve at those bits | The pack | Below the curve | Bits saved at the same KL |
|---|---|---|---|---|---|
| Trellis K = 4 | 4.125 | 0.0480 (UD-IQ4_XS to UD-Q4_K_XL) | 0.0398 | 17% | 0.40 |
| Trellis K = 3 + 4, `down` | 3.458 | 0.0685 (UD-IQ3_XXS to UD-Q3_K_XL) | 0.0619 | 10% | 0.18 |
| Trellis K = 3 + 4, `freqcc` | 3.459 | 0.0685 | 0.0664 | 3% | 0.06 |
| Trellis K = 3 | 3.125 | 0.0916 (UD-Q2_K_XL to UD-IQ3_XXS) | 0.0765 | 16% | 0.17 |
| P3 (seeds) | 4.000 | 0.0509 | 0.0600 | 18% above | |
| P4 (seeds) | 4.500 | 0.0403 | 0.0408 | 1% above | |

Unsloth's expert sizes, from the earlier page and as measured here:

| Pack | Earlier page | Here |
|---|---|---|
| UD-IQ3_XXS | 49.6 GB, 3.28 bits | 50.29 GB, 3.331 bits |
| UD-Q3_K_XL | 55.7 GB, 3.69 bits | 56.18 GB, 3.720 bits |
| UD-IQ4_XS | 59.9 GB, 3.97 bits | 60.38 GB, 3.999 bits |
| UD-Q4_K_XL | 77.0 GB, 5.10 bits | 77.02 GB, 5.101 bits |

## Notes

- `down` beats `freqcc` at the same bits (0.0619 against 0.0664). The `down_proj` tensors matter more than the most
  used experts. The seed hybrids showed the same order (0.0533 for `down`, 0.0540 for `freqcc`).
- At 2.1 and 2.5 bits the KL rises steeply: 0.204 and 0.164. No measured Unsloth pack is that small. UD-IQ2_XS,
  UD-IQ1_M, UD-IQ1_S and UD-Q2_0 are not measured: `tools/hf_llama_kld.sh` expects 4 shards with the PLE table in
  the last one, and these packs have 3. So the 2-bit packs have no curve to compare with yet.
- UD-Q2_K_XL's perplexity (3.773) is below BF16's (3.802), but its KL is 0.0994. KL is the measure here.
- The size correction moves Unsloth's packs 0.03 to 0.05 bits to the right. It moves the curve up at a given number of
  bits, so earlier comparisons against it favour Unsloth a little. The seed pages' conclusions keep their direction.
- Both 2-bit packs (32 GB and 37 GB) fit a 12 GB GPU and 32 GB of RAM (plan T6). Their quality
  cost is large, so a mix with more K = 3 may be the better trade there.
