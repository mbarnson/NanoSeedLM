# Qwen3.8-Flash-Next on CUDA: one-pass prompt attention, a faster DeltaNet step, and the same quality (2026-11-05)

**Result.** A 29,120-token prompt of the P3 pack went from 3,423 to 3,660 tokens per second on an RTX PRO 6000. On an
A100 it went from 1,691 to 1,879. Prompt attention now runs in one pass with an online softmax: 1,362 to 926 ms at
29,120 tokens. Against the 4-bit MLX model on 4,096 tokens, P3's KL divergence is 0.161 now and 0.164 with the engine
of the day before. The expert prefetch made prompts below about 3,000 tokens slower. It now waits for larger forwards.

**Status.** Current. One prompt text, one pack. The RTX 4080 is not measured.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell Server Edition, 96 GB), and `a100-large` (A100 80 GB) for
  one row. Each row is a different job and node. The load average was 1 to 8 on the RTX PRO 6000 nodes and 6 to 9 on
  the A100 nodes. Each prompt speed is the mean of two runs, and the two runs agree within 2%.
- Pack: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) at
  `5753a67`, an 8 GB expert cache (3,255 slots).
- Prompts and decode as in [merged GEMMs](2026-11-05-qwen38-merged-gemms.md). The prefetch sweep used the first 600 to
  17,000 characters of the wikitext-2 test set (`scratchpad/rowsjob/pfsweep.sh`).
- Quality: `tools/hf_qwen4_kld.sh` with `TOKENS=4096`, the reference
  [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp) at
  `825df62`, and a 48 GB cache. It ran at `364b9ad` (the engine of
  [the packs against the 4-bit model](2026-11-04-qwen38-kld-vs-4bit.md)) and at `fa1d190`.

## Results

The 15,922-token prompt and decode on the RTX PRO 6000:

| Commit | Change | Prompt, tokens/s | Decode, tokens/s |
|---|---|---|---|
| `ff17acd` | Start ([merged GEMMs](2026-11-05-qwen38-merged-gemms.md)) | 3,377 | 54.2 |
| `d525c44` | The HC add after attention and the next mix's norm in one kernel (the same bits) | 3,370 | 53.7 |
| `5f56c70` | Attention: one pass, an online softmax, P V on mma.sync | 3,523 | 54.2 |
| `fa1d190` | DeltaNet: g and beta for a stage first, two chains for each dot product | 3,679 | 53.1 |

The 29,120-token prompt:

| GPU | `470d42a` | `5f56c70` | `fa1d190` |
|---|---|---|---|
| RTX PRO 6000 | 3,423 | 3,576 | 3,660 |
| A100 | 1,691 (`ff17acd`) | 1,866 | 1,879 |

Quality on the first 4,096 tokens of wikitext-2 (KL of the 4-bit reference against P3):

| Engine | Mean KL | Median KL | 99th percentile KL | Top-1 agreement | Perplexity, reference | Perplexity, P3 |
|---|---|---|---|---|---|---|
| `364b9ad` | 0.1644 | 0.0140 | 1.98 | 90.01% | 1.823 | 1.829 |
| `fa1d190` | 0.1608 | 0.0141 | 2.12 | 90.38% | 1.807 | 1.838 |

Prompt speed with the expert prefetch and without it (`Q4E_NO_PREFETCH=1`), one forward each, tokens per second:

| Prompt tokens | `fa1d190`, on | `fa1d190`, off | `b93125c`, on |
|---|---|---|---|
| 149 | 83 | 220 | 223 |
| 502 | 286 | 518 | 524 |
| 1,089 | 613 | 876 | 881 |
| 2,082 | 1,170 | 1,296 | 1,299 |
| 3,930 | 2,193 | 1,903 | 2,120 |

GPU time by stage at `fa1d190` (the 29,120-token prompt, RTX PRO 6000):

| Stage | ms | Share |
|---|---|---|
| Experts | 2,844 | 37.9% |
| Projections | 1,423 | 19.0% |
| Hyper-connections | 1,164 | 15.5% |
| Attention | 926 | 12.3% |
| Gated DeltaNet | 495 | 6.6% |
| QSA select | 276 | 3.7% |
| Norms, RoPE and stores | 174 | 2.3% |
| PLE | 136 | 1.8% |
| Router and cache | 59 | 0.8% |

## Notes

- The quality figures move in both directions: the reference's perplexity fell 0.9% and P3's rose 0.5%. Mean KL fell
  2% and the top-1 agreement rose 0.4 points. We read this as rounding and the odd QSA near-tie, not a loss. It is
  one text of 4,096 tokens.
- The one-pass kernel scales a row's O and sum by exp(old max - new max) at each tile. O sits in mma.sync fragments,
  and their layout is documented. A lane holds rows g and g + 8, so the scale is a register multiply. The wmma API
  does not document its fragment layout, so S stays on wmma and P V moved to mma.sync.
- `5f56c70` and `fa1d190` changed the greedy decode text again (165, 150, then 164 tokens): the first tokens are near
  ties. The kernel and engine tests pass with the same error figures.
- The prefetch admits all 512 experts of a layer. A short prompt uses fewer of them, and its layers compute too
  briefly to hide the copies. `b93125c` prefetches only for forwards of 64 router picks an expert or more (about 3,300
  rows for P3). The 128-row logits forwards of `nslm-qwen4-kld` had become 2 times slower, and are back.
- The engine test now also runs `tests/data/qwen4_tiny_attn`, a tiny model with strong attention, against the C
  reference. On it, zero keys move the reference's logits by 1.7 (of 1.7).
- At 29,120 tokens, pc-cuda's GEMM kernels (experts and projections) take 57% of the time.
