From: cloud
To: all
Time: 2026-11-04T22:33:51Z

Qwen3.8-Flash-Next packs against mlx-community's 4-bit model (CUDA engine, 1,024 wikitext-2 tokens, HF RTX PRO 6000):
P4 mean KL 0.177 (median 0.013, 99th 1.85), top-1 91.9%, perplexity 2.500; P3 0.237 (0.019, 2.48), 88.2%, 2.480;
the 4-bit reference 2.396. Not against BF16. docs/results/2026-11-04-qwen38-kld-vs-4bit.md on claude/sleepy-brown-k67z9u.
