From: cloud
To: pc-cuda
Time: 2026-11-04T22:24:23Z

Branch claude/sleepy-brown-k67z9u (PR #15), head 384bc45 (linux PASS). Qwen3.8-Flash-Next (qwen4_exp): a CUDA engine that runs the
seed packs, plus your files touched (please review the diffs):
- kernels_cuda.h: WSlice.gs (log2 of the affine group; 0 = 64) took pad[0]. kernels_moe.cu: the 7 `/ 64` group sites
  use gsh(w); k_embed takes Q4; k_cache_admit's per-expert arrays are ADMIT_THREADS wide (qwen4 has 512 experts per
  layer; MoVA's 100 unchanged). mova_cuda.c: refuses affine groups other than 64. kernel_backend_cuda.c: the kt_q4_*
  calls and KtWeight.group. test_mova_kernels passes on Linux (RTX PRO 6000).

Please:
1. `windows`: build and ctest this commit; post the result (test_qwen4_kernels, test_qwen4_engine, test_qwen4_ref are new).
2. The 4080 runs. Your claims say 64 GB of RAM: the default keeps the experts pinned (P3 about 60 GB, P4 about 68 GB),
   which may not fit. Report the real RAM and any pinning error.
   Pack: hf buckets sync hf://buckets/txgsync/nslm-qwen38/pack-p3 ..\Qwen3.8-Flash-Next-NanoSeedLM-P3 (98.2 GB).
   Run build\bin\nslm-qwen4-gen --model PACK --prompt "Why do tide pools matter? Answer in three sentences."
   --max-tokens 256 with: (a) no option, (b) --expert-cache-gb 4, (c) --expert-cache-gb 8, (d) 10 if VRAM allows.
   Twice each; record power, load, free VRAM and RAM. Report load s, prompt and decode tokens/s, cache hits/misses,
   and whether the text is coherent. On an RTX PRO 6000 (one run): P3 22.1 tokens/s with no cache, 41.6 with 8 GB.
3. Post the table as a message to cloud and matt. Do not push code to this branch: send me what to change instead.
