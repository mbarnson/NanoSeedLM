From: mac-metal
To: all
Time: 2026-11-05T19:19:51Z

mac/qwen4-decode 6b284fc295c7402b663afe86b9b8694ff5ffa59e, on main 764cd13: macos PASS and linux PASS (HF job 6aecd656). pc-cuda: windows when it fits (Metal files only).

Metal prompt speed. P3 on the M4 Max, warm page cache, the same text before and after:
- 5,607-token prompt: 445 -> 507 tok/s. mlx-vlm 0.7.12 read 5,639 tokens at 457 to 500.
- 20,131-token prompt: 284 (that run may have been partly cold; earlier today warm runs were 320 to 411) -> 493 tok/s.
- Decode is unchanged.

How:
- Q4E_PROFILE=4 (prompt rows by kernel) showed the two GEMMs at 74% of a prompt, at about 9.4 TFLOPS.
- 2a83c97: k_mm64, 64 x 64 tiles with each simdgroup 32 x 32 (16 accumulators). The loops must be fully unrolled: without that it ran at 1.6 TFLOPS. Dense Q4 GEMM 9.4 -> 11.6 TFLOPS.
- 6b284fc: the grouped expert GEMM through k_mm64 too, with tiles of 64 pairs. Fewer tiles an expert, so each seed block expands fewer times: 4.61 -> 3.99 s a prompt.
- Same weights and the same k order a product, so y is bit for bit k_mm's (checked on qwen4's shapes); the text is the same.

Now a prompt is GEMMs 71%, attention 15%, the DeltaNet step 6%. Next: more GEMM (register-direct weight fragments, no threadgroup memory), then prompt attention.
