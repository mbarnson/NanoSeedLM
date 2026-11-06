# Packs as shipped: the 4-bit MLX dense weights triple the KL (2026-11-05)

**Result.** With the 4-bit MLX dense weights that P3 ships, the KL against BF16 is 0.182 for P3 and 0.185 for the
trellis mix. With Unsloth's Q8_0 dense weights it is 0.0600 and 0.0619. So the dense weights add about 0.12, three
times the KL of the experts. Top-1 agreement falls from 91.5% to 84%. Unsloth's 3-bit packs keep their dense weights
at Q6_K or Q8_0, and the release packs need dense weights of that precision too.

**Status.** Current. Step D0 of the [dense plan](../plans/2026-11-05-trellis-dense.md): hypothesis HD1 holds.

## Setup

- KL: llama.cpp against BF16, 64 windows of 2,048 tokens of wikitext-2, as on
  [the trellis mixes page](2026-11-05-trellis-mixes.md). The PLE table is BF16's for every pack.
- Job: HF Jobs `rtx-pro-6000x4`, `6aed6158c78863f9c633db32` at `7eb5d9b`, 25 minutes, about $4.56. It saved the BF16
  reference to `/b/kld-ref/ref-c64.kld` for later jobs.
- Packs: `nslm-qwen4-emulate --dense` writes the dense tensors of P3's folder (4-bit MLX, group 32) into Unsloth's
  Q8_0 shards, with the experts of each pack. That is 664 tensors: DeltaNet, attention, shared experts,
  hyper-connections, `lm_head` and the embedding. The routers, norms, `conv1d` and the indexer stay Unsloth's.
- Before the job, rows of each kind of tensor were read over HTTP from P3 and from Unsloth's shards. With
  llama.cpp's order of DeltaNet's value heads, they agree within 0.08 to 0.10, the error of 4-bit weights.

```bash
hf jobs run --flavor rtx-pro-6000x4 -v hf://buckets/txgsync/nslm-qwen38:/b -e SRC=/b/src-7eb5d9b -e OUT=/b/kld-d0 \
    -e TQBANK=/b/tq -e MAPS=/b/maps -e REF=/b/kld-ref -e "PACKS=Q8_0 TQ3L12+TQ4L12@down/mlx P3/mlx" \
    nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /b/src-7eb5d9b/tools/hf_llama_kld.sh
```

## Results

| Pack | Dense weights | Mean KLD | Median KLD | 99th percentile KLD | Top-1 agreement | Perplexity |
|---|---|---|---|---|---|---|
| Q8_0 | Q8_0 | 0.0199 | 0.0051 | 0.228 | 95.23% | 3.813 |
| P3 (earlier page) | Q8_0 | 0.0600 | | 0.723 | 91.55% | |
| P3 as shipped | 4-bit MLX | 0.1819 | 0.0702 | 1.772 | 84.07% | 4.194 |
| Trellis mix (earlier page) | Q8_0 | 0.0619 | 0.0169 | 0.755 | 91.44% | |
| Trellis mix as shipped | 4-bit MLX | 0.1851 | 0.0694 | 1.850 | 84.01% | 4.210 |

BF16's perplexity is 3.802. Q8_0 reproduced its KL of the earlier pages (0.01986). The standard error of each mean KLD
is 0.0003 to 0.0015.

The dense types of Unsloth's packs, from their GGUF headers:

| Kind | UD-IQ3_XXS | UD-Q3_K_XL |
|---|---|---|
| DeltaNet projections | Q6_K, Q8_0 | Q6_K, Q8_0 |
| Hyper-connections | Q8_0 | Q8_0 |
| Shared experts | Q8_0, Q6_K | Q8_0, Q6_K |
| `lm_head` | Q6_K | Q6_K |
| Embedding | Q6_K | IQ4_XS |

## Notes

- The dense weights are 4 GB of the pack, and each token reads all of them. The routed experts are 47 to 62 GB, and a
  token reads 10 of 512 in each layer. So the dense weights need more bits than the experts for the same KL.
- The 4-bit MLX weights are rounded to the nearest level, with no imatrix. Unsloth's types use an imatrix.
- The published P3 and P4 cards give a KL against the 4-bit MLX model, not against BF16. As shipped, P3's KL against
  BF16 is 0.182.
- MLX 8-bit dense weights (`mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp`) read 1.7 times the bytes of the 4-bit ones.
  Trellis-coded dense weights (steps D1 to D4) may get the quality of 8 bits with fewer bytes.
