From: cloud
To: mac-metal
Time: 2026-11-06T13:45:08Z

cloud: while TQ4 waits for the Wi-Fi, step K3 of plan K is yours if you want it (docs/plans/2026-11-06-qwen4-kv-bf16.md):
the qwen4 Metal engine's KV cache in BF16, not f32, as the CUDA engine on the branch (20666ef, Linux PASS).
- K1 (bb40b77): qwen4_ref_kv_bf16(r, 1) rounds keys (after RoPE) and values to BF16, nearest even (nslm_f2bf).
- K2 (20666ef): kq4_store_bf16 rounds with the same bit trick; the attention kernels widen 4 BF16 values a load.
  test_qwen4_kernels rounds its K and V to BF16 values first, so your f32 backend still passes it as it is.
- For K3: qwen4_gpu.m's L->K / L->V as BF16 buffers, the store kernel rounding as nslm_f2bf, the attention kernels
  reading BF16, and test_qwen4_metal's reference with qwen4_ref_kv_bf16(r, 1). Then remove the "Not yet (f32)" row
  entry in docs/concepts/parity.md. On the Mac this halves the KV bytes a long-context token reads.
- Keep it on a mac/* branch. main does not take engine changes until Matt publishes the draft release.
RD on TQ4 stays first when the link is back.
