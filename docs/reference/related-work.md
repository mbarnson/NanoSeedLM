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

- [QTIP](https://arxiv.org/abs/2406.11235) and
  [ik_llama.cpp trellis quants](https://github.com/ikawrakow/ik_llama.cpp/pull/113). The kernel computes the weights.
  It does not look them up in a table.

## Discussion

- [Hacker News discussion of SeedLM](https://news.ycombinator.com/item?id=43599967).
