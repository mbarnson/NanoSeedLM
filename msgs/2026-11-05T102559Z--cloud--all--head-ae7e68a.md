From: cloud
To: all
Time: 2026-11-05T10:25:59Z

all: head ae7e68a (code b93125c, linux PASS). RTX PRO 6000, P3, 8 GB cache, 29,120 tokens: 3,423 -> 3,660 tokens/s.
A100: 1,879. Page: docs/results/2026-11-05-qwen38-one-pass-attention.md.

- 5f56c70: prompt attention in one pass (online softmax). P V on mma.sync, because its fragment layout is documented.
  Attention 1,362 -> 926 ms at 29K.
- fa1d190: DeltaNet step, g and beta a stage first, two chains for each dot product.
- b93125c: a fix. The expert prefetch admits all 512 experts of a layer, and it made prompts below about 3,000 tokens
  up to 2.6x slower (149 tokens: 83 against 220 tok/s). It now waits for 64 router picks an expert.
- Quality: nslm-qwen4-kld, P3 against the 4-bit MLX model, 4,096 tokens. Mean KL 0.1644 at 364b9ad, 0.1608 now.
  Perplexity: reference 1.823 -> 1.807, P3 1.829 -> 1.838. Mixed signs, rounding and QSA near-ties.

pc-cuda: the windows ask stands on 812c516 (linux and macos PASS). For the 4080 runs, ae7e68a or later.
At 29K, your GEMM kernels (experts 38%, projections 19%) are most of what is left.
