Status: PASS
Branch: main
Commit: 6364ffb6a6c9793e83a1416df7244c9a318a7ee2
Platform: linux
Agent: pc-cuda
Time: 2026-10-09T01:11:54Z

Linux: HF Jobs, nvidia/cuda:12.8.1-devel-ubuntu24.04 (gcc 13, CMake 3.28, CUDA 12.8.93; architectures 80-real 86-real 89
120-real), clean build (1 warning). Two runs: RTX PRO 6000 Blackwell Server Edition (job 6ac83ead095c578089300855) and H200
(job 6ac83eaffee2c90070171d27). ctest 18/18 passed on both. Skipped: test_affine, test_mova_cfg and test_tokenizer (no
model folder in the job). GPU tests ran (test_engine, test_mova_kernels, search GPU tests).
Build dependency: harness/tokenizer.c needs ICU (unicode/unorm2.h). CMake does not check for it, so without libicu-dev
the build fails at compile time, not at configure. Install libicu-dev (+ pkg-config). Worth a find_package(ICU) in
CMakeLists.txt (shared file: whoever touches it next).
