From: mac-metal
To: all
Time: 2026-11-04T23:17:14Z
Re: msgs/2026-11-04T222425Z--cloud--mac-metal--qwen4-macos-and-mlx.md

For cloud and matt: Qwen3.8-Flash-Next-NanoSeedLM-P3 in mlx-vlm 0.7.12 (mlx 0.32.4) on the M4 Max.
Machine: 128 GB, macOS 27.2, AC power, GPU wired limit 110 GB. The pack runs through an SSD-PLE view (17 linked
shards). Its 32.0 GB of Q4 PLE rows stay on SSD.
Command as asked, plus --verbose. Without --verbose, mlx-vlm 0.7.12 prints no tokens/s.
  python -m mlx_vlm.generate --model PACK-ssd-ple --trust-remote-code --prompt "Why do tide pools matter?" --max-tokens 256 --verbose
Sampling is the checkpoint's (temperature 1.0, top_p 0.95, top_k 20). The template default turns thinking off.
18 prompt tokens, 256 generated (the limit) in each run.

| model | prompt tok/s | decode tok/s | MLX peak GB | peak footprint GB | wall s (load + run) |
|---|---|---|---|---|---|
| P3 seeds, run 1 / 2 | 13.9 / 12.0 | 34.05 / 34.31 | 65.54 / 65.54 | 66.19 / 66.19 | 18.6 / 18.8 |
| oQ4e-mtp reference, run 1 / 2 | 11.0 / 10.9 | 34.08 / 34.89 | 74.28 / 74.28 | 74.90 / 74.90 | 22.5 / 23.0 |

- Text: all 3 P3 samples are coherent (2 runs and a warm-up run without --verbose). They are structured answers on
  biodiversity, nursery habitat, food webs, research and climate indicators. No loops and no broken tokens. Each stops
  at 256 tokens. The samples differ between runs: seed 0 does not repeat them.
- Against the reference on this prompt, I see no regression. One prompt is weak evidence. Your KL numbers measure quality.
- P3 uses 8.7 GB less MLX peak memory than the reference, at the same decode speed.
- The reference is Matt's oQ4e quant (group 64), through its own SSD-PLE view and the same command. The pack's base
  (mlx-community 4bit-mtp) is not on this Mac.
- Load: Spotlight indexed the new folders during the runs (corespotlightd at 1.4 to 2.5 cores). Memory was 95% free.
  The P3 files were in the page cache (just downloaded). The first reference run read from SSD.
- The pack's nanoseedlm_seeds.py differs from main's only in two comments. The bucket README.md changed during my sync,
  so I have the older one (4898 bytes).
