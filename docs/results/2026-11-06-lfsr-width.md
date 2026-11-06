# Wider LFSR seeds and the state stride: weight error on Qwen expert rows (2026-11-06)

**Result.** Each extra seed bit multiplies the error of P = 3 and P = 4 blocks by about 0.83, from K = 16 to 22. So
P = 3 with a 20-bit seed has the error of P = 4 at the same 4.5 bits. A decode that reads every 4th LFSR state gives 8%
to 10% less error, with no extra bits and no extra search time. No seed can match real weights exactly at P = 3 sizes,
because a BF16 weight carries about 10.5 bits.

**Status.** Current. Weight error without activation weights, on 2,048 blocks of one layer. Read
[Search objectives and KL on SmolLM2-135M](2026-11-06-search-objective-kl.md) for the KL.

## Setup

- Machine: a cloud container with 4 CPU cores (AVX-512) and 15 GB of RAM, and no GPU. The load average was 0.1 before
  the runs, with no other jobs. The page gives no speed results.
- Rows: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision `de4b8e4`, layer 20,
  experts 0 to 3. From each expert, 8 rows of `gate_up_proj` (2,560 inputs) and 8 rows of `down_proj` (640 inputs),
  read with HTTP range requests.
- Blocks: 1,536 blocks of 8 weights from `gate_up_proj` and 512 from `down_proj`, spread evenly over the rows.
- Search: the search of `nslm/lib_searchp.c` in a probe. It has an f32 Cholesky, the exponents e0, e0 - 1 and e0 + 1,
  and the 3^P refit by decoded BF16 error. It uses no activation weights, and it does not clamp the exponent window.
- Generator: a K-bit Fibonacci LFSR with primitive taps, found by a check of the period. Each state gives its top 16
  bits, so the decode arithmetic does not change. At K = 16 the probe uses the repository's taps. It matches
  `lfsr_states` for all 65,535 seeds.
- Error: sum (w - w')^2 / sum w^2 over the blocks.
- Code: `tools/seed_kl_probe/` at the commit of this page. `lfsr_width.c` is the scalar probe. `seedlib.c` is the
  vectorized search, which `check_lib.py` checks against `lfsr_width.c`.

```bash
mkdir -p out/data && python3 -I tools/seed_kl_probe/fetch_qwen_rows.py out/data
clang -O3 -march=native -ffp-contract=off -pthread tools/seed_kl_probe/lfsr_width.c -o out/lfsr_width -lm
out/lfsr_width 3 lfsr 18 out/data/gu.f32 out/data/dn.f32 2048            # P = 3, K = 18
STRIDE=4 out/lfsr_width 4 lfsr 16 out/data/gu.f32 out/data/dn.f32 2048   # P = 4, every 4th state
QB=5 out/lfsr_width 3 lfsr 16 out/data/gu.f32 out/data/dn.f32 2048       # P = 3, 5-bit coefficients
out/lfsr_width 3 rand 18 out/data/gu.f32 out/data/dn.f32 2048            # a random codebook of 2^18 entries
clang -O2 -Inslm tools/seed_kl_probe/walsh.c -o out/walsh -lm && out/walsh out/data/gu.f32 16
clang -O2 -Inslm tools/seed_kl_probe/bm.c -o out/bm && out/bm out/data/gu.f32 4096
```

## Results

Error by seed width. The search time doubles with each seed bit. The P = 3 search of all Qwen experts at K = 16 took
83 minutes on 4 RTX PRO 6000 (about $15.60). Read [the seed search](2026-11-04-qwen38-seed-search.md).

| Seed bits K | P = 3 bits per weight | P = 3 error | P = 4 bits per weight | P = 4 error | Search time against K = 16 |
|---|---|---|---|---|---|
| 16 | 4.000 | 0.01449 | 4.500 | 0.00680 | 1x |
| 17 | 4.125 | 0.01214 | 4.625 | 0.00547 | 2x |
| 18 | 4.250 | 0.00977 | 4.750 | 0.00454 | 4x |
| 19 | 4.375 | 0.00815 | 4.875 | 0.00377 | 8x |
| 20 | 4.500 | 0.00676 | 5.000 | 0.00323 | 16x |
| 21 | 4.625 | 0.00565 | 5.125 | 0.00270 | 32x |
| 22 | 4.750 | 0.00477 | 5.250 | 0.00224 | 64x |

On 512 of the blocks, P = 3 gives 0.01424 at K = 16, 0.00679 at K = 20 and 0.00340 at K = 24.

The same bits, spent in three ways:

| Bits per block | Bits per weight | P = 3, wider seed | P = 4, wider seed | P = 3, 5-bit coefficients |
|---|---|---|---|---|
| 35 | 4.375 | 0.00815 (K = 19) | | 0.01088 (K = 16) |
| 36 | 4.5 | 0.00676 (K = 20) | 0.00680 (K = 16) | 0.00900 (K = 17) |
| 38 | 4.75 | 0.00477 (K = 22) | 0.00454 (K = 18) | |
| 40 | 5.0 | about 0.0035 (K = 24) | 0.00323 (K = 20) | |

The codebook and the state stride, at K = 16 unless the row gives a size:

| Codebook | P = 3 error | P = 4 error |
|---|---|---|
| LFSR, every state (the format) | 0.01449 | 0.00680 |
| LFSR, every 2nd state | 0.01320 | 0.00645 |
| LFSR, every 3rd state | 0.01328 | 0.00635 |
| LFSR, every 4th state | 0.01308 | 0.00624 |
| LFSR, every 8th state | 0.01296 | 0.00625 |
| LFSR, every 16th state | 0.01316 | 0.00626 |
| Random 16-bit states, 2^16 entries | 0.01294 | 0.00623 |
| Random 16-bit states, 2^18 entries (LFSR K = 18: 0.00977, 0.00454) | 0.00884 | 0.00418 |
| Random 16-bit states, 2^20 entries (LFSR K = 20: 0.00676, 0.00323) | 0.00602 | 0.00288 |

Limits, on the same rows:

| Measurement | Value |
|---|---|
| Error of Q8_0 (blocks of 32) | 2.86e-5 |
| Seed bits to reach the error of Q8_0, at 0.83 for each bit (an extrapolation) | about 33 more: K = 49, 8.1 bits per weight |
| Entropy of one BF16 weight: sign 1.00, exponent 2.55, mantissa for a given exponent 6.91 bits | 10.46 bits |
| Linear complexity of 4,096 BF16 weights (65,536 bits), by Berlekamp-Massey | 32,768, half the bits |
| Linear complexity of the 16-bit LFSR's own stream (the control) | 16 |
| Walsh terms of the error of a P = 3 block at a fixed (q, e), 16 blocks | 452 to 529 of 65,536 |

## Notes

- A seed bit costs 1/8 bit for each weight. An error factor of 0.83 for each seed bit is about 6.6 dB for each bit per
  weight. The rate-distortion slope is 6.0 dB. So seed bits are a little better than average bits. They do not close
  the gap of 0.83 to 0.90 bits between the seeds and the bound for Gaussian weights.
- With real-valued coefficients, the best subspace at K = 16 leaves 62% of the P = 3 error (0.00902). For P = 4 it
  leaves 33% (0.00222). The int4 coefficients cause the rest.
- A wider seed beats wider coefficients. At 4.375 bits it gives 25% less error than 5-bit coefficients.
- Adjacent LFSR states are correlated: state k + 1 is about state k / 2 plus a new top bit. The random codebook shows
  that this costs 8% to 11% of the error. A stride of 2 recovers most of it, and a stride of 4 recovers all of it.
- With a stride of 2, one 64-bit window gives the 24 states of a P = 3 block. The window is the 16-bit seed and a
  48-bit table entry. A stride of 4 needs a window of 112 to 144 bits. No kernel was written or measured.
- A K-bit seed names at most 2^K blocks. At 10.5 bits for each weight, a block of 8 weights has about 2^84 likely
  values. P = 3 stores 32 bits. So an exact P = 3 code exists for a block with a probability of about 2^-52. A faster
  search, quantum or classical, does not change this count.
- Berlekamp-Massey finds the shortest LFSR that gives a bit string exactly, in polynomial time. For the weights it is as
  long as half the string, as for random bits. This is also true for the sign and exponent bits alone.
- Fix q and e. Then the error of a P = 3 block is a constant plus a few hundred terms v (-1)^<b, s>, with s the seed
  bits. The probe `walsh.c` checks this against direct evaluation for all seeds, to 5e-16, and its minimum is the winning
  seed. This is the form of a weighted max-XORSAT problem and of a diagonal Hamiltonian of Pauli Z products. Read
  [the quantum plan](../plans/2026-11-06-quantum-seed-search.md).
- Matt asked whether a wider LFSR would improve the KL and perplexity of P3 and P4. The answer for weight error is yes,
  at about the rate of any bit, and at an exponential search cost.
