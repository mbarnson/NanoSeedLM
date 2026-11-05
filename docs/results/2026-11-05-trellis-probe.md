# Trellis-coded experts against ggml's types and seeds: weight error (2026-11-05)

**Result.** A bitshift trellis with QTIP's computed code and imatrix-weighted Viterbi search beats ggml's types at 3 and 4
bits for each weight. At 3.11 bits its weighted error is 0.0158, against 0.0182 for IQ3_S at 3.44 bits. At 4.11 bits it
is 0.0042, against 0.0057 for IQ4_XS at 4.25 bits. At 2.12 bits it equals IQ2_S at 2.56 bits (0.0609 against 0.0602).
The trellis saves about 0.4 to 0.5 bits for each weight at the same error, and it is close to the bound for Gaussian
weights. Our P = 3 and P = 4 seeds are 0.6 to 0.9 bits from that bound.

**Status.** Current. Weight error only, on a sample of rows of one layer. Step T1 of the
[trellis plan](../plans/2026-11-05-trellis-experts.md). Two corrections to the text below (2026-11-05, from the
page's own errors and output files). The P = 3 and P = 4 seeds are 0.83 to 0.90 bits from the bound, not 0.6 to 0.9.
At L = 16 the LFSR code has 0.6 to 1.7% more weighted error than 1MAD, not "within 1%".

## Setup

- Rows: the rows of the [rotation probe](2026-11-04-qwen38-rotation-probe.md). Layer 20, experts 0 to 3, 8 rows of
  `gate_up_proj` (2,560 inputs) and 32 rows of `down_proj` (640 inputs) from each expert.
- Weights: Unsloth's imatrix, the mean squared input of each column for each expert. The error is
  sum h e^2 / sum h w^2, with e the weight error and h the column's weight.
- Trellis (`tq.c`, a probe outside the repository):
  - Blocks of 256 weights (`gate_up_proj`) and 320 (`down_proj`), each with a 16-bit scale.
  - L-bit states and K bits for each weight.
  - The code c(s) is the sum of the 4 bytes of 34038481 s + 76625530 (mod 2^32), normalized.
  - Three passes of Viterbi search, each with a refit scale.
- Seeds: the P = 4 fits of the rotation probe, and the activation-weighted P = 3 search on the same rows.
- ggml: llama.cpp `5ce06fc`, `ggml_quantize_chunk` with the same imatrix, on an HF CPU job (`6aecf4f8`).

## Results

`gate_up_proj`, weighted error:

| Bits | ggml type | Error | Bits | NanoSeedLM | Error |
|---|---|---|---|---|---|
| 2.06 | IQ2_XXS | 0.1134 | 2.12 | Trellis, L 16, K 2 | 0.0609 |
| 2.31 | IQ2_XS | 0.0860 | 3.10 | Trellis, L 12, K 3 | 0.0179 |
| 2.56 | IQ2_S | 0.0602 | 3.11 | Trellis, L 14, K 3 | 0.0166 |
| 2.63 | Q2_K | 0.0698 | 3.11 | Trellis, L 16, K 3 | 0.0158 |
| 3.06 | IQ3_XXS | 0.0313 | 4.00 | P3 (seeds) | 0.0136 |
| 3.44 | IQ3_S | 0.0182 | 4.11 | Trellis, L 16, K 4 | 0.0042 |
| 3.44 | Q3_K | 0.0207 | 4.50 | P4 (seeds) | 0.0064 |
| 4.25 | IQ4_XS | 0.0057 | | | |
| 4.50 | Q4_K | 0.0049 | | | |
| 4.50 | IQ4_NL | 0.0056 | | | |

`down_proj`, weighted error. ggml's 2- and 3-bit types need rows of a multiple of 256 weights, and these rows have 640:

| Bits | Format | Error |
|---|---|---|
| 2.09 | Trellis, L 16, K 2 | 0.0595 |
| 3.09 | Trellis, L 16, K 3 | 0.0154 |
| 4.00 | P3 (seeds) | 0.0131 |
| 4.09 | Trellis, L 16, K 4 | 0.0041 |
| 4.50 | P4 (seeds) | 0.0062 |
| 4.50 | IQ4_NL | 0.0055 |
| 4.50 | Q4_0 | 0.0065 |

## Notes

- For Gaussian weights, the lowest possible error at R bits is 2^-2R: 0.0625 at 2 bits, 0.0156 at 3 and 0.0039 at 4.
  Weighted search adds a few percent of gain on top. The trellis reaches about the bound at the bits it stores, and
  its overhead (the first state and the scale) is 0.09 to 0.12 bits.
- Gate TA asks for IQ2_S's error at about 2.1 bits and IQ3_S's at about 3.1 bits. It passes at 3.1 bits and misses by
  1% at 2.1 bits. Hypothesis HT1 (0.4 bits fewer for the same error) holds.
- The code from our LFSR stream (the byte sum of 32 stream bits) gives the same error within 1%.
- The encoder's speed on one core, in weights a second: 172,000 at L 12, 40,000 at L 14 and 9,000 at L 16.
- ggml's 2- and 3-bit types cannot quantize `down_proj` here, so llama-quantize must use a type with 32-weight blocks
  there (4.5 bits or more). We did not check Unsloth's tensor types. A trellis can use 2 to 4 bits there.
- Weight error does not give the KL. Step T2 measures the KL of a whole model.
