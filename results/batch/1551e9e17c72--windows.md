Status: FAIL
Branch: batch
Commit: 1551e9e17c723ca661e53aceb469e48fef779751
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T00:01:58Z

Windows 11, MSVC + CUDA 13.1, RTX 4080, clean build in build-v (Release). ctest: 17/18 passed, test_affine skipped.
FAIL: test_engine, in two cases only:
  FAIL: MLA fp8 batch: eng_open: KV format 2 not supported by the CUDA engine (MLA: bf16; GQA: bf16, q8)
  FAIL: MLA fp4 batch: eng_open: KV format 3 not supported by the CUDA engine (MLA: bf16; GQA: bf16, q8)
Everything else passed, bit-exact: the MLA expanded / blocks-of-16 prefill, batching and slot copies; test_mova_kernels
(the new kt_* stubs print "not in this backend"); test_kvq; test_serve_splitter; test_kv_disk.
Cause: test_batch(dir, "MLA fp8"/"MLA fp4", ...) at tests/test_engine.c:551-552 does not skip on an engine that
refuses the format, unlike the single-sequence fp8/fp4 loop at :475. My CUDA fp8/fp4 port makes it pass.
Linux: not covered yet.
