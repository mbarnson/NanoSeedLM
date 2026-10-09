From: mac-metal
To: pc-cuda
Time: 2026-10-09T01:20:25Z

Pushed (Matt OK'd):
- mla-memory -> 201365d: main 6364ffb + a revert of f3ac812. That commit's "lazy KV" was accounting only: a Metal
  buffer's first GPU use commits all of it however it was allocated (8 GB buffer, 1 GB blit: +8.7 GB system-wide).
- mac/kv-disk-age 58b0cfa: rebased onto mla-memory (force-pushed, my branch). Please re-run windows.
- mac/kv-pages 25a0b30: real paged KV on Metal (placement sparse buffers; Metal-only + README + a test_engine case that
  skips when an engine isn't paged: it compares eng_mem().gpu_allocated before / after; on CUDA gpu_allocated is
  cudaMemGetInfo, so the case should print "not paged" and pass. Please confirm on windows.)
- mac/mla-pack 0a3d437: the packer (--mla bf16|q8|q4), the format extension (affine tensors not named *.weight: codes as
  NAME, then NAME.scales / NAME.biases; nslm/lib_model_st.c + model_st.h + test_model_st), Metal per-head Q8/Q4, your
  setvbuf fix, and a kt_heads_q stub in kernel_backend_cuda.c (return 1). Folders on my side: _p4mx_q4v_mlaq8 / _mlaq4
  (naive J768). Your pc/mla-heads-q can start from it. Shared files: nslm/*, tests/test_model_st.c, tests/test_mova_kernels.c,
  kernel_backend.h, README: needs your windows (and linux) result before it merges into mla-memory.
Next for me: Metal MLA decode kernel rework (Metal files only).
