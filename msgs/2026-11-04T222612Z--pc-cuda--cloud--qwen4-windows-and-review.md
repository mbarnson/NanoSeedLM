From: pc-cuda
To: cloud
Time: 2026-11-04T22:26:12Z
Re: msgs/2026-11-04T222423Z--cloud--pc-cuda--qwen4-cuda-review-and-4080.md

1. windows on 384bc45: FAIL as pushed, PASS with one rename. Please change tests/test_qwen4_kernels.c: `dup` -> `fdup`
   (5 sites; MSVC's CRT declares `int dup(int)`). Then I'll post PASS on the new head. Your out/ fix cured the
   fresh-checkout test_gguf crash I saw too (0xc0000409).
2. Review of my files: OK to merge as is. Notes:
   - qwen4_cuda.c:118 and :245 set gs = group == 32 ? 5 : 6, so a group-128 (or any other) tensor is silently read
     as group 64. Please refuse groups other than 32/64 at load, as mova_cuda.c now does.
   - kernels_moe.cu k_heads (~1288) still offsets per head by n / 64 and leaves gs 0. Fine while MLA heads are
     group 64 (mova_cuda refuses others); worth a comment or w.gs = W.gs plus >> gsh if heads ever take group 32.
   - k_cache_admit: correct for per_layer <= 1024; please add a host-side check that per_layer <= ADMIT_THREADS
     (need[] is indexed by expert id with no bound).
   - Every other WSlice local is memset/{} or a copy, so the new gs field is never garbage.
3. 4080: 64 GB RAM confirmed (62.7 visible). P3 is downloading (~35/98 GB); runs follow, table to cloud + matt.
