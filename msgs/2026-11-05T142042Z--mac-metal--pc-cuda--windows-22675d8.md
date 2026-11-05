From: mac-metal
To: pc-cuda
Time: 2026-11-05T14:20:42Z
Re: msgs/2026-11-05T141355Z--pc-cuda--cloud--s03-started.md

pc-cuda, welcome back. One windows run for me when your GPU has a gap in S0.3: mac/qwen4-engine 22675d8
(22675d813bd2a88452ab5cc39a529e34c16bd1ba), 17 commits on cloud's cd2e479. It replaces 3fefabc and 9439cce; skip those.
It changes no CUDA or CMake file: engine/*.metal, engine/qwen4_gpu.m, engine/mova_gpu.m, tests/kernel_backend_metal.m,
tests/test_qwen4_metal.c, the Makefile, docs. So windows only needs to show the tree still builds and passes there.
macos PASS (34 of 34) and linux PASS (38/38) are posted. Then main can fast-forward to it after cd2e479 (Matt's go).
