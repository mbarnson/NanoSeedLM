# The prompt time of Qwen3.8-Flash-Next on the CUDA engine, by stage (2026-11-04)

**Result.** On an RTX PRO 6000 with an 8 GB expert cache, 1,024 prompt tokens of the P4 pack took 15.4 s. Two
changes bring this to 7.5 s, 2.1 times faster. The PLE prefetch cuts the PLE stage from 5,415 ms to 4 ms. The grouped
expert GEMMs cut the expert stage from 3,820 ms to 1,118 ms. The router and the expert cache now take 57% of the time.

**Status.** Current. The test used one text of 1,024 tokens on HF Jobs, not the RTX 4080. The numbers are GPU stream
times from `Q4E_PROFILE=1`.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell, 96 GB), `tools/hf_qwen4_kld.sh`. Each job ran on a different
  node. The load average at the start was 8.22 (baseline), 5.66 (prefetch) and 1.23 (grouped GEMMs).
- Models: [Qwen3.8-Flash-Next-NanoSeedLM-P4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4), and
  the reference mlx-community/Qwen3.8-Flash-Next-4bit-mtp at `825df62`. Both run on the CUDA engine.
- Text: the first 1,024 tokens of the wikitext-2 test set, in prompt rows of 128.
- Command, with `Q4E_PROFILE=1`:

```bash
nslm-qwen4-kld --ref /ref --model /model --text /text.txt --tokens 1024 --expert-cache-gb 8
```

| Run | Commit | Job |
|---|---|---|
| Baseline | `5ed8d1b` | `6aebbe5e0e0338d1bb8046c5` |
| PLE prefetch | `8ba5c38` on `5ed8d1b` (rebased: `68c66bc`) | `6aebc2bbc78863f9c63309b3` |
| Grouped expert GEMMs | `5134f0f` | `6aebc4720e0338d1bb8048ee` |

## Results

GPU stream time of the P4 pack for 1,024 tokens, in ms:

| Stage | Baseline | PLE prefetch | Grouped expert GEMMs |
|---|---|---|---|
| PLE | 5,414.6 | 4.1 | 4.3 |
| Router and expert cache | 4,207.1 | 4,207.7 | 4,225.7 |
| Experts | 3,816.8 | 3,820.0 | 1,117.7 |
| Hyper-connections | 755.8 | 750.2 | 752.0 |
| Head | 693.5 | 708.4 | 727.3 |
| Gated DeltaNet | 315.3 | 316.0 | 317.4 |
| Attention | 205.4 | 205.3 | 206.0 |
| Embedding | 0.3 | 0.6 | 0.6 |
| **Total** | **15,409** | **10,012** | **7,351** |

Wall time for 1,024 tokens, and the quality against the reference:

| Run | P4, s | Reference, s | KL mean | KL median | Top-1 | Perplexity, reference / P4 |
|---|---|---|---|---|---|---|
| Baseline | 15.4 | 13.0 | 0.17879 | 0.01476 | 90.04% | 2.364 / 2.478 |
| PLE prefetch | 10.2 | 8.1 | 0.17879 | 0.01476 | 90.04% | 2.364 / 2.478 |
| Grouped expert GEMMs | 7.5 | 7.8 | 0.17601 | 0.01319 | 90.23% | 2.402 / 2.502 |

## Notes

- **PLE.** Each token reads 16 random rows of the 32 GB PLE table. Before the prefetch, the CPU waited for each row,
  one page fault at a time, while the GPU had no work. `q4e_forward` now asks the OS to read all rows of the call first
  (`madvise` WILLNEED). The logits did not change.
- **Experts.** The gather matvecs read an expert's weights, and expand the P4 seeds, once for each (token, expert)
  pair. The grouped GEMMs read them once for each tile of up to 128 pairs. The reference's affine experts gain less
  (1,274 ms to 910 ms), because they need no seed expansion.
- **Quality cost.** The grouped GEMMs round their inputs to BF16, as the dense GEMMs do. This moves the logits of both
  models. The reference's perplexity on this text went from 2.364 to 2.402 (1.6%), and P4's from 2.478 to 2.502. The
  KL between them went down from 0.179 to 0.176. One text cannot tell a real loss from the rounding noise. A
  comparison with mlx-vlm's float32 logits on more text can.
- **Noise.** The stages that do not wait for the CPU agree within 1% across the three nodes. So the different loads did
  not change the result.
- **Next.** The router and expert cache stage is the copy of missing experts over PCIe. With 128 rows a forward, most
  experts of a layer miss in each forward. Larger prompt rows copy each expert fewer times, but take VRAM from the cache.
- The RTX 4080 decode profile (pc-cuda, `5ed8d1b`, P3, 8 GB cache) put 17% of the decode time in the PLE stage. The
  prefetch may also help decode. A rerun on the 4080 will tell.
