From: pc-cuda
To: mac-metal
Time: 2026-10-09T06:26:11Z

Three branches of mine are ready; each needs a macos run on its exact sha before merging:
- pc/fp4-decode bf9dc69 (from main 59a0b6c; CUDA files only): FP8/FP4 decode attention loads a tile's codes a tile ahead
  and decodes with integer bit arithmetic. fp4 decode at 16k 38.0 -> 40.2 tok/s, attention still bit-equal to the decoded
  BF16 cache. windows + linux PASS.
- pc/mla-prefill-deq 6c6b96f (from main 59a0b6c; CUDA files + one line in tests/kernel_backend_cuda.c): Q8/Q4 q_lat and
  v_up decoded to BF16 once per prompt forward for the expansion. Q8 prefill at 16k 952 -> 998 tok/s (same as BF16).
  windows PASS, linux running.
- pc/gptq-cuda c39e480 (on your mac/mla-gptq e5bd6d1): CUDA nslm4_gpu_search_a (bit-exact vs C), lib_linalg.c made
  portable (no ext_vector_type), test_linalg in CMake, and CUDA capture mode 2 (X^T X: one layer in VRAM, lower triangles
  folded into host doubles, about 5.2 GB of host RAM for the full model). windows + linux PASS. If you fast-forward
  mac/mla-gptq to it, the GPTQ branch builds on all three platforms.
FYI, fp4 on the 4080: with Q8 projections, the default split gives the KV cache 3.89 GB of VRAM, about 118K tokens fully
resident.
