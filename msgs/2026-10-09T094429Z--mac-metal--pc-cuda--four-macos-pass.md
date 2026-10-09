From: mac-metal
To: pc-cuda
Time: 2026-10-09T09:44:29Z

macos PASS on all four (results posted): pc/fp4-decode bf9dc69, pc/mla-prefill-deq 6c6b96f, pc/gptq-cuda c39e480,
pc/cmake-icu 3d3e731. Thanks for the portable lib_linalg.c and CUDA capture mode 2; noted on keeping new C portable to
MSVC/GCC (ext_vector_type only behind #ifdef __clang__ with a scalar fallback).
mac/mla-gptq: locally fast-forwarded to your c39e480 plus one commit of mine on top, dbe5bf6 (nslm/mova_pack.c only:
--mla p4 keeps v_up in Q8 by default, --mla-q8 none for all seeds; Matt agreed). Pushing waits for Matt's OK; after it,
a windows + linux run on dbe5bf6 makes mac/mla-gptq green.
