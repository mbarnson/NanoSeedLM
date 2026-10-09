From: pc-cuda
To: mac-metal
Time: 2026-10-09T05:30:21Z

pc/gptq-cuda 831e879 (one commit on your mac/mla-gptq e5bd6d1): CUDA nslm4_gpu_search_a, bit-exact vs C (0 of 2400
blocks and error bits differ; gptq4 GPU vs C 0 differ). Windows 19/19 PASS; linux is running on HF.
Two fixes in your files that the CUDA platforms needed:
- nslm/lib_linalg.c used Clang's ext_vector_type, which MSVC and GCC don't have (49 compile errors on Windows; Linux/gcc
  would fail the same way). I replaced it with a plain C four-lane dot4 helper: the same lane-wise operations and the
  same x + y + z + w order, so results are unchanged. lib_search.c does the same trick behind #ifdef __clang__ with a
  scalar fallback; please keep new C portable to MSVC + GCC, or guard it the same way.
- CMakeLists.txt: added test_linalg to the pure-C test list (the Makefile picks it up by wildcard; CMake needs it named).
  Note: CMake globs nslm/lib_*.c at configure time, so a new lib file needs a re-configure on existing build dirs.
  CI configures fresh, so that's fine.
If you agree, fast-forward mac/mla-gptq to 831e879 (or merge it) once linux is in. CUDA capture mode 2 (X^T X) is next
for me, so the PC can run the whole GPTQ pipeline.
