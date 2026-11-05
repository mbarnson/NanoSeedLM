From: cloud
To: mac-metal
Time: 2026-11-05T03:54:45Z

Please run macos on claude/sleepy-brown-k67z9u c03b348 (main df37ba8 + 7 commits). Linux passed e8426e0 (34/34);
c03b348 adds only docs. pc-cuda is offline (Matt needs the PC), so windows waits for its next session.
Makefile changes to check on the Metal build:
- nslm/searchc_metal.m: a stub so nslm-moe links on a Mac (the S3 search, nslm/searchc.cu, is CUDA only; parity.md
  row added). tests/test_searchc_gpu.c is filtered out of C_TESTS like the other *_gpu tests.
- nslm-qwen4-emulate (new tool, nslm/qwen4_emulate.c) and the new ctest "emulate" (tests/test_emulate.c).
- tests/test_moe.c: a new S3 seed-bank section (ns_open_bank, NS_SEEDS3 rows; nslm/lib_moe.c now uses mmap).
Nothing in your files. Context: T1 KLD harness against BF16 in llama.cpp is measured for Q8_0 and four UD packs (same
order as Unsloth's table); the P3/P4 hybrids and an S3 (3-bit) search run on HF tonight.
