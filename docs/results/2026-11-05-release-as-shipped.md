# The release packs as shipped: KL with 8-bit dense weights (2026-11-05)

**Result.** With 8-bit MLX dense weights, the three release packs have a KL against BF16 of 0.0415 (TQ4), 0.0638
(TQ3-D4) and 0.0778 (TQ3). That is 2% to 4% above their KL with Unsloth's Q8_0 dense weights. With P3's 4-bit MLX
dense weights it was 0.17 to 0.19. Gate RB of the [release plan](../plans/2026-11-05-release-family.md)
passes. TQ3 gives the same text in two runs on CUDA. The runs of TQ3-D4 and TQ4 did not start (a script error), so
step R5 checks them.

**Status.** Current. Steps R2b, R3 and R4 of the release plan.

## Setup

- Folders: `tools/hf_release_pack.sh`, `nslm-qwen4-pack --tq --dense8`, in the bucket at `/b/release/`. Each folder
  holds P3's tensors without its seed experts and the pack's trellis banks of layers 0 to 48. The dense modules are
  `mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp`'s (8-bit, its groups of 128 written as two groups of 64).
- KL: llama.cpp against the saved BF16 reference, 64 windows of 2,048 tokens of wikitext-2, as on
  [the trellis mixes page](2026-11-05-trellis-mixes.md). `nslm-qwen4-emulate` writes each folder's experts and dense
  tensors into Unsloth's Q8_0 shards. The PLE table is Unsloth's Q8_0.
- Runs: `nslm-qwen4-gen --expert-cache-gb -1 --no-think --max-tokens 128`, a code prompt, two runs for each folder, on
  an RTX PRO 6000.
- Jobs: HF Jobs `rtx-pro-6000x2`. `6aed6ff3` packed TQ3 (23 minutes). `6aed75a2` packed TQ3-D4 and TQ4 and tested all
  three (48 minutes, about $4.39).

## Results

| Pack | Folder | Expert bits | Mean KLD | Median KLD | 99th percentile KLD | Top-1 agreement | Perplexity |
|---|---|---|---|---|---|---|---|
| TQ4 | 95 GiB | 4.125 | 0.0415 ± 0.0005 | 0.0113 | 0.492 | 92.98% | 3.847 |
| TQ3-D4 | 86 GiB | 3.458 | 0.0638 ± 0.0007 | 0.0175 | 0.765 | 91.27% | 3.861 |
| TQ3 | 81 GiB | 3.125 | 0.0778 ± 0.0008 | 0.0217 | 0.939 | 90.40% | 3.865 |
| Q8_0 (control) | | 8.5 | 0.0199 | 0.0050 | 0.232 | 95.16% | 3.809 |

The same experts with other dense weights:

| Pack | Unsloth's Q8_0 dense | 8-bit MLX dense (this page) | 4-bit MLX dense |
|---|---|---|---|
| TQ4 | 0.0398 | 0.0415 | 0.1734 |
| TQ3-D4 | 0.0619 | 0.0638 | 0.1851 |
| TQ3 | 0.0765 | 0.0778 | 0.1909 |

BF16's perplexity is 3.802. Q8_0 reproduced the KL of the earlier pages with Q8_0's PLE table (0.01991).

## Notes

- TQ3's two runs gave the same text (md5 26deda07): correct Python, with a docstring.
- The runs of TQ3-D4 and TQ4 found no `nslm-qwen4-gen`: `tools/hf_llama_kld.sh` rebuilds `/w/src` between packs. The
  script now keeps its binaries in `/w/bin`. Step R5 downloads and runs the packs on the RTX 4080 and the M4 Max.
- A memory map of the bucket mount faults with SIGBUS, so the script tests a local copy of each folder.
- Unsloth's packs keep their dense weights at Q6_K or Q8_0 (see [packs as shipped](2026-11-05-dense-as-shipped.md)).
  Their KLs on the trellis mixes page are 0.0509 (UD-IQ4_XS, 3.999 bits), 0.0593 (UD-Q3_K_XL, 3.720) and 0.0735
  (UD-IQ3_XXS, 3.331).
