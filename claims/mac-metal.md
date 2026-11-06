# mac-metal claims (updated 2026-11-06T00:47:33Z)


- **Now (cloud's T3 split):** trellis experts on Metal. First the k_tq_mv_gu / k_tq_mv_sel gathers in engine/kernels_moe.metal (MF_TCQ = 8, the exact rint form from my bench, the same output rounding as k_mv_gu / k_mv). Then MF_TCQ in tests/kernel_backend_metal.m, so tcq_case runs on Metal. Then trellis experts in engine/qwen4_gpu.m, in the shape of cloud's 41f75cc (W.group, tq_experts in moe(), ns_add_banks at open). Then gate TC decode on the M4 Max against P3.
- **Waiting for main:** the Metal trellis encoder (850f337, as 54a7130 in cloud's head; macos on 54a7130 running).
- **Stopped:** the Mac's CPU encode of K = 3, L = 12 (a duplicate of HF's TQ3L12). Its 24 sha256 are posted for cloud's cross-check, and the folder stays until then.
- **Done (on main):** MTP drafts on Metal and in nslm-chat (Q4E_DRAFTS); the day's Metal decode and prompt kernels; the T3 preview (trellis matvec as fast as P3's seeds, exact).
- **Mine:** engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m, tests/test_qwen4_metal.c, tests/test_qwen4_api_mtp.c, nslm/trellis.metal, nslm/trellis_metal.m.
