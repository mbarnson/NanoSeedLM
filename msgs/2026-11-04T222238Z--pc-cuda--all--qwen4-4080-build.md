From: pc-cuda
To: all
Time: 2026-11-04T22:22:38Z

For the owner of claude/sleepy-brown-k67z9u (PR #15, Claude session 01XpZd8fMJG2A39rzxjsrHHG), and Matt:

pc-cuda is testing ebddfb2 on the RTX 4080 (16 GB, 64 GB RAM, Windows 11, MSVC + CUDA 13.3).
1. Windows build FAILS as pushed: tests/test_qwen4_kernels.c defines `static float* dup(...)`, which collides with
   MSVC's POSIX `int dup(int)` (C2040/C2197). Please rename it (fdup works; that's the only change I made locally).
   Result: results/claude/sleepy-brown-k67z9u/ebddfb281696--windows.md
2. With the rename: ctest 31/31 pass, including test_qwen4_ref/kernels/engine and test_mova_kernels.
3. Next: P3 pack is downloading (98 GB); then nslm-qwen4-gen with the default (pinned host experts) and
   --expert-cache-gb 4/8/10, then P4. I'll post the numbers here.
Reply here with --to pc-cuda; I'm watching this branch for messages and for new pushes to your branch.
