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

## Discussion

- [Hacker News discussion of SeedLM](https://news.ycombinator.com/item?id=43599967).
