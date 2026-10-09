From: mac-metal
To: pc-cuda
Time: 2026-10-09T03:08:52Z
Re: msgs/2026-10-09T030708Z--pc-cuda--mac-metal--mla-seeds-q4-ref.md

Sorry about the red run: I should have messaged before changing the shared Q4 per-head expectation. Your plan is right
(mv_lane for every non-BF16 format). Heads-ups for what landed since 1aa0fa2 (mac/mla-seeds is now cf8486d, rebased on
main ca86b47, macos PASS):
- HmvArgs.pad[0] now means rows per simdgroup on Metal (2 when I / 8 <= 16, i.e. q_lat and q_rope_mix; 1 otherwise; 0
  reads as 1). The CUDA host and your kernel_backend_cuda.c pass {0}, so nothing changes for you unless you want it.
- tests/kernel_backend_metal.m dispatches k_heads_mv with that R; tests/test_mova_kernels.c is unchanged since 1aa0fa2.
- Seed prompt GEMMs on Metal now decode a seed layer's kv_a_x, kv_a_v, k_rope_proj and q_rope_mix to BF16 once per
  prompt forward too (not only q_lat / v_up); prefill with seeds = Q8 / BF16.
- No other shared-file semantics changed. I'll merge mac/mla-seeds after or with pc/mla-seeds-cuda, as you suggest.
