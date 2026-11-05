# Related work

## The methods that NanoSeedLM uses

- [SeedLM](https://arxiv.org/abs/2410.10714), Shafipour et al., ICLR 2025. Weights as seeds of a pseudo-random
  generator. Read [SeedLM](../concepts/seedlm.md).
- [TransMLA](https://arxiv.org/abs/2502.07864), Meng, Yao and Zhang, 2025. Conversion of GQA to MLA. Read
  [Multi-head latent attention](../concepts/mla.md).

## Other work on seeds

- [Seed-Q](https://arxiv.org/abs/2609.38477). Sensitivity-aware LFSR seed quantization.
- [AWSRC](https://arxiv.org/abs/2608.23144). Seeded residual coding on top of INT4.
- [SeedLM+O](https://github.com/ropaes1/seedlm-o-results). A measured negative result on small dense models.
- [seedlm-repro](https://github.com/qiuyu-ren/seedlm-repro). A NumPy replication.
- [PRANC](https://arxiv.org/abs/2206.08464) and [NOLA](https://arxiv.org/abs/2310.02556). Networks and adapters from
  random bases.

## Weights that the kernel computes

- [QTIP](https://arxiv.org/abs/2406.11235), Tseng, Sun, Hou and De Sa, NeurIPS 2024. A bitshift trellis over a
  computed Gaussian code, with a randomized Hadamard transform. The kernel computes the weights. It does not look them
  up in a table. Read the [trellis plan](../plans/2026-11-05-trellis-experts.md).
- [ik_llama.cpp trellis quants](https://github.com/ikawrakow/ik_llama.cpp/pull/113). Trellis-coded formats (IQ*_KT)
  with CPU and CUDA kernels in a llama.cpp fork.

## Trellis-coded quantization

- [Trellis coded quantization of memoryless and Gauss-Markov sources](https://doi.org/10.1109/26.46532), Marcellin
  and Fischer, IEEE Transactions on Communications, 1990. The origin of TCQ: a Viterbi search over a trellis, within
  0.21 dB of the distortion-rate bound for a uniform source.
- [QuIP](https://arxiv.org/abs/2307.13304), Chee, Cai, Kuleshov and De Sa, NeurIPS 2023. Incoherence processing and
  adaptive rounding (LDLQ): the first usable 2-bit LLMs.
- [QuIP#](https://arxiv.org/abs/2402.04396), Tseng, Chee, Sun, Kuleshov and De Sa, 2024. A randomized Hadamard transform,
  E8 lattice codebooks and BlockLDLQ. QTIP replaces the lattice with a trellis.
- [BCJR-QAT](https://arxiv.org/abs/2605.10655), Iyengar, 2026. The trellis search made differentiable (BCJR
  forward-backward) for quantization-aware training.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3). Its EXL3 format is a variant of QTIP, in an inference library
  for consumer GPUs.
- [CCQ](https://arxiv.org/abs/2507.07145), Baidu, 2025. Convolutional codes with scalar quantization at 2.0 to 2.75 bits,
  with bit-shift decoding and no codebook table. Tested on DeepSeek-V3 and ERNIE-4.5-300B-A47B (MoE models).

## Trellis codes from recursions and shift registers

We found no other work that uses an LFSR or SeedLM's seeds as the code of a trellis. These are the nearest.

- [The design of trellis waveform coders](https://doi.org/10.1109/TCOM.1982.1095515), Stewart, Gray and Linde, IEEE
  Transactions on Communications, 1982. The decoder is a shift register and a function of its contents. A bitshift
  trellis has this structure.
- [Linear congruential trellis source codes: design and analysis](https://doi.org/10.1109/TCOMM.2007.904375), Eriksson,
  Anderson and Goertz, IEEE Transactions on Communications, 2007. A trellis quantizer whose values come from a linear
  congruential recursion. A 2005 letter by Anderson and Eriksson reports 0.24 dB from the bound for a Gaussian source
  at 1 bit, with 512 states. These are the ancestors of QTIP's computed codes.
- [Viterbi-based efficient test data compression](https://doi.org/10.1109/TCAD.2011.2172609), Lee and Roy, IEEE
  Transactions on Computer-Aided Design, 2012. The Viterbi algorithm, not linear equations, picks the inputs of a test
  decompressor. It is the trellis form of a seed search.
- [Viterbi-based pruning](https://iclr.cc/virtual/2018/poster/256), Lee, Ahn, Kim, Chuang and Kim, ICLR 2018, and
  [Double Viterbi](https://iclr.cc/virtual/2019/poster/1103), Ahn, Lee, Kim and Kim, ICLR 2019. Viterbi encoders of
  XOR gates and flip-flops give sparse and quantized weights of CNNs and LSTMs.
- [Structured compression by weight encryption](https://arxiv.org/abs/1905.10138), Kwon et al., 2019, and
  [encoding weights of irregular sparsity](https://arxiv.org/abs/2105.01869), Park et al., 2021. XOR-gate decoders for
  pruned and quantized weights.

## Discussion

- [Hacker News discussion of SeedLM](https://news.ycombinator.com/item?id=43599967).
