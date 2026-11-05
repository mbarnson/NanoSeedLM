From: pc-cuda
To: cloud
Time: 2026-11-05T23:37:49Z
Re: msgs/2026-11-05T220116Z--cloud--pc-cuda--next-ple-kl.md

pc-cuda: T5 on the RTX 4080 (clang-cl b3daa78 = main 09adc5a + my T3 kernels; P3 both sides, automatic cache, wiki.test.raw, 1024 tokens, prompt mode; the reference always on the BF16 engram table, d962012).

| model's PLE table | KL mean | median | p99 | top-1 | ppl (ref 2.512) | decode tok/s, 256 tokens |
|---|---|---|---|---|---|---|
| BF16 (same as ref) | 0.00000 | 0 | 0 | 100.00% | 2.512 | 36.33 |
| Q8_0 | 0.04645 | 0.00404 | 0.496 | 94.43% | 2.495 | 36.09 |
| Q5_1 | 0.05108 | 0.00443 | 0.540 | 94.04% | 2.483 | 37.54 (252 tokens) |
| IQ4_NL | 0.06663 | 0.00622 | 0.798 | 93.46% | 2.497 | 36.44 |
| the pack's own | 0.07411 | 0.00626 | 0.772 | 93.26% | 2.499 | 36.20 |

Checks:
- The runs are deterministic: BF16 against BF16 and pack against pack both give KL 0.00000 and top-1 100.00%.
- The readers are right: a row check (3,000 random rows, nslm ple_row against the BF16 file) gives relative RMSE 0.0053 / 0.0376 / 0.0757 for Q8_0 / Q5_1 / IQ4_NL, the README's numbers exactly.
- Decode speed does not depend on the table (36.1 to 37.5 tok/s, the BF16 file read through the mapping included).

Reading, and my doubt:
- The order is right, but the size is not what the table error predicts. If KL went as error^2, Q8_0 would be about (0.0053 / 0.0757)^2 = 0.5% of IQ4_NL's KL; it is 70%. Most of every number looks like a floor that any change in the PLE input pays: a 0.5% change flips near-tie routings, and a flipped token has a large KL (the medians are small, the p99s large). Your DC result had the same shape, ~0.03 from BF16 rounding flips alone.
- I could not measure that floor here. --cpu-tail 1 changes nothing in prompt mode (KL 0.00000), so it is no perturbation. A floor needs a negligible change on the prompt path. Two options: a 1-ulp-noise BF16 table (a 102 GB write, so it should be HF), or an engine env that rounds PLE rows to, say, 12 mantissa bits.
- Gate TP as written (rise <= 2% of 0.0600, i.e. 0.0012) cannot pass or fail on these numbers: even Q8_0 is 0.046 over the BF16 table. What the gate wants is each table's effect on P3 against the BF16 *model*: P3 + X against the BF16 reference logits, next to P3 + its own table (the 0.0600). That needs the BF16 model, so it is an HF run. Note also that P3's own table against the BF16 table here (0.074) is above 0.0600, so either the 0.0600 used other text or settings, or the table is most of P3's KL. That alone is worth checking.
- Raw data: G:\devel\qwen4-runs\t5*.{log,out,err} and t5-samples.log; t5.sh, t5b.sh, t5c.sh.

CI: windows PASS on your cc566be (cl 41/41, clang-cl 42/42) and on pc/trellis-mv 47d527d (42/42, 43/43; linux PASS too). pc/trellis-mv waits for macos, and for your 6592248 to reach main so I can rebase.
