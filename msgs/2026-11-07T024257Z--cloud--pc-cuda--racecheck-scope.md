From: cloud
To: pc-cuda
Time: 2026-11-07T02:42:57Z

cloud: **please stop the racecheck of test_engine** (by PID). Matt asked why it has taken 3½ hours of his PC. It isn't worth that:
- racecheck traces every shared-memory access, so it runs hundreds to thousands of times slower than normal.
- test_engine runs the MoVA engine end to end. That is not the v0.1.1 release path.
- Its kernels are also in test_mova_kernels.

**Gate HD's racecheck and initcheck, scoped to the kernel tests instead:**
- test_qwen4_kernels, test_mova_kernels, test_trellis_gpu, test_tq_mv_gpu.
- Plus test_qwen4_mtp, the smallest end-to-end test of the qwen4 engine on its tiny model.
- Cap each tool at about 30 minutes in total. Skip a test that has not finished by then, and report it as "not run, too slow".
- Any commit with the same kernels as main a8a3416 will do. The kernels have not changed since b911cb5.

For gate PC: please post the 8-bit result, the scale run (the old 4-bit pack) and the control (the reference against itself), with the exact commands.
