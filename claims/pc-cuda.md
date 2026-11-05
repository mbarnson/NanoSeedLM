# pc-cuda claims (updated 2026-11-05T02:03:51Z)

- **Now (2026-11-04, Matt's go):** the fp4 MLA decode gap on CUDA, branch `pc/fp4-decode-gap` from main db0ea15.
  Decode rows at positions >= 12288 (NSLM_MLA_DEC_LONG) take 32 heads a block and 64 splits, so each FP4 key is decoded
  once for all heads; shorter rows keep today's path. Files: engine/kernels_moe.cu, engine/kernels_cuda.h,
  engine/mova_cuda.c (mine); **tests/test_mova_kernels.c (shared)**: two latent-attention cases with 40 splits and
  MlaArgs.pad[2] = 150. Metal ignores pad[2] for MLA attention, so on macOS they run the normal split path.
- Queue (CUDA files): SEED6P8 reader for the CUDA engine; Windows VRAM reserve default; KV VRAM rows by active slot;
  full-model GPTQ capture when Matt is around.
