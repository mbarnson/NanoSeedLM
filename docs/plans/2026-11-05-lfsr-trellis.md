# Plan: an LFSR code for the trellis (2026-11-05)

**Goal.** Find whether the trellis keeps its quality and speed when its code comes from NanoSeedLM's LFSR, in place
of QTIP's 1MAD hash. If it does, the packs stay as SeedLM made them: no stored codebook, and the LFSR as the
generator.

**Status.** Draft. Matt asked on 2026-11-05 whether SeedLM has a future beside the trellis. L2 is done: the LFSR
code is within 0.6% of 1MAD at K = 3, and 1.5% higher for `down_proj` at K = 4
([the probe](../results/2026-11-05-lfsr-trellis-probe.md)).

## Why

- The trellis packs replace every seed expert, and they are ahead on each axis that we measured (table below).
- The trellis already computes its code from a pseudo-random function of its state. Only the generator differs from
  SeedLM's.
- At L = 12 the LFSR code (the byte sum of 32 LFSR stream bits) had the weighted error of 1MAD within 1%. At L = 16
  it had 0.6% to 1.7% more. Read [the trellis probe](../results/2026-11-05-trellis-probe.md).
- At L = 12 a code has only 4,096 values. An engine can expand any generator into a table of 8 KB when it loads a
  pack. Then the decode speed does not depend on the generator. The LFSR is linear, so two tables of 64 values each
  also give its 32 bits (`G[s] = G[lo] ^ G[hi]`).

| | P4 seeds | Trellis K = 4 | P3 seeds | Trellis mix (K = 3, `down_proj` at K = 4) |
|---|---|---|---|---|
| Bits for each expert weight | 4.50 | 4.125 | 4.00 | 3.46 |
| KL against BF16 | 0.0408 | 0.0398 | 0.0600 | 0.0619 |
| Decode on the RTX 4080, tokens per second | | | 42.2 | 54.4 |

The weighted error of the seeds is 0.83 to 0.90 bits from the bound for Gaussian weights. The trellis is about 0.1
bits from it.

## Steps and gates

| Step | Work | Who | Gate |
|---|---|---|---|
| L1 | A table code: q(s) from a table of 2^L values in the C reference, the three encoders and the decode kernels | cloud, pc-cuda, mac-metal | LA |
| L2 | The LFSR table, and the weighted error on the probe's rows at L = 12, K = 3 and K = 4 | cloud (CPU) | LB |
| L3 | Trellis K = 3 of the whole model with the LFSR table, and its KL against BF16 | HF (encode, KL) | LC |
| L4 | K = 4 and the mix with the LFSR table, and decode on the RTX 4080 and the M4 Max | HF, pc-cuda, mac-metal | LD |

- **LA.** The 1MAD table gives the same bytes as the computed code, on every encoder and kernel. Decode is not
  slower on either engine.
- **LB.** The weighted error of the LFSR code is within 1% of 1MAD's at K = 3 and K = 4.
- **LC.** The KL of LFSR K = 3 is within 2% of the KL of 1MAD K = 3 (0.0765).
- **LD.** The same for K = 4 and the mix, and decode within 2% of 1MAD's on both machines. Then the format takes the
  LFSR code (format v1), and the release packs use it.

## Hypotheses

| ID | Hypothesis | Gate |
|---|---|---|
| HL1 | A table code decodes at least as fast as the computed code on both engines | LA |
| HL2 | At L = 12 the LFSR code gives the KL of 1MAD within 2% | LC |

## Costs

- L1: kernel work on both engines. A pack needs one table of 2^L values, 8 KB at L = 12, in fast GPU memory.
- L2: the CPU of the cloud container.
- L3: an encode on 8 RTX PRO 6000 (about 5 minutes, about $2) and a KL job with the saved BF16 reference (about $1.50).
- L4: two encodes and a KL job, about $6.

## What seeds may still do better

- We know of no measured axis. Each trellis weight reads its own L-bit window, so any weight decodes alone, as in a
  seed block. We did not compare the encode cost directly. The trellis encoded all experts in about 5 minutes for
  each K on 8 GPUs.
- An LFSR needs only shifts and XORs, and 1MAD needs a 32-bit multiply. Hardware without a fast multiplier favours an
  LFSR code. That is a case for an LFSR trellis, not for seed blocks.

## See also

- [Trellis-coded experts](2026-11-05-trellis-experts.md), the plan for the format.
- [Trellis-coded experts](../concepts/trellis.md), the concept page.
