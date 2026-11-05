# pc-cuda claims (updated 2026-11-05T23:37Z)

- **Online** (Claude Code CLI session). Matt gave the PC; no run over about an hour.
- T3 CUDA: pc/trellis-mv 47d527d (k_tq_mv_gu / k_tq_mv_sel, kc_tq_*, test_tq_mv_gpu, tcq_case in test_qwen4_kernels, MF_TCQ = 8 in kernels_moe.metal, nslm-mova-kbench rows). windows and linux PASS; waiting for macos, and for cloud's 6592248 on main to rebase.
- T5: done, posted (msgs ...--pc-cuda--cloud--t5-ple-formats-4080.md). The gate needs an HF run against the BF16 model.
- Parked: pc/trellis-cuda 82882e2 (in cloud's branch as 32257cf).
- Done this session: windows PASS on cb3d6b6, e816fe8, bc50f41, 7dd456a, 09adc5a, 41a8fe6, cc566be; D4 on the 4080 (--cpu-tail 1 and 2 lose).
