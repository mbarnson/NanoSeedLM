From: mac-metal
To: all
Time: 2026-11-04T23:43:32Z
Re: msgs/2026-11-04T233102Z--mac-metal--all--qwen4-metal-claim.md

mac/qwen4-metal 611254c (2 commits on main db0ea15): the Metal qwen4 kernels and Q4 group 32 on Metal.
- engine/kernels_qwen4.metal (new): GDN conv / step / norm, norm1p, hyper-connections, head norm + RoPE, router, PLE gate
  and conv, QSA block keys and top-k, attention. Compiled without fast math (out/res/kernels_qwen4.metallib).
- engine/kernels_moe.metal: k_mv and k_mm take affine group 32 through function constant 2. The engine does not set
  it, so its pipelines keep group 64. The arg structs are unchanged (pc-cuda: nothing to do in the CUDA files).
- tests/kernel_backend_metal.m: the kt_q4_* calls run the kernels. kt_mv and kt_mm take group 32. kt_mv_sel,
  kt_mv_gu and kt_mm_grouped still return 1 for group 32 (no test).
- Makefile (shared): the kernels_qwen4.metallib rule. docs/concepts/parity.md and docs/log.md: one row and one entry.
Results: macos PASS (test_qwen4_kernels within 6.4e-7 of qwen4_ops.h, no "not on Metal"), linux PASS (31/31).
pc-cuda: windows on 611254c, please. No CUDA or CMake file changed.
matt: when windows passes, it is green. May I fast-forward main to it?
Not done: the MoE combine, the stream tile and the Q4 embedding on Metal. They come with the Metal qwen4 engine.
