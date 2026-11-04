Status: FAIL
Branch: claude/sleepy-brown-k67z9u
Commit: ebddfb28169674dae92720c72a744c7abcdb4836
Platform: windows
Agent: pc-cuda
Time: 2026-11-04T22:22:26Z

Build fails on MSVC (VS 2022 + CUDA 13.3, RTX 4080, driver 610.88):
  tests/test_qwen4_kernels.c(20): error C2040: 'dup': 'float *(const float *,size_t)' differs in levels of indirection from 'int (int)'
  (then C2197 'int dup(int)': too many arguments, at lines 38, 67, 136, 166)
MSVC's CRT declares POSIX dup(int) (io.h, pulled in via the headers). Renaming the test's static dup() to fdup()
(5 occurrences, local only, not pushed) fixes it.
With that rename: ctest 31/31 passed x3 (test_qwen4_ref, test_qwen4_kernels, test_qwen4_engine, test_mova_kernels
all pass). Skipped: test_affine, test_mova_cfg, test_tokenizer (no model dir). One intermittent crash: test_gguf exit
0xc0000409 on the first run right after the build; it didn't repeat in 13 later runs.
