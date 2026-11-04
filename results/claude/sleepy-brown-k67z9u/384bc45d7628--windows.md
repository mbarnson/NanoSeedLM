Status: FAIL
Branch: claude/sleepy-brown-k67z9u
Commit: 384bc45d762878909db1a950a17955070a527d89
Platform: windows
Agent: pc-cuda
Time: 2026-11-04T22:26:07Z

As pushed: build fails on MSVC (VS 2022, CUDA 13.3, RTX 4080): tests/test_qwen4_kernels.c(20) error C2040 'dup' vs the
CRT's int dup(int); C2197 at lines 38, 67, 136, 166.
With only `dup` -> `fdup` in that file (local): ctest 31/31 PASS from a fresh out/ (test_qwen4_ref, test_qwen4_kernels,
test_qwen4_engine, test_mova_kernels, test_gguf all pass). Skipped: test_affine, test_mova_cfg, test_tokenizer.
