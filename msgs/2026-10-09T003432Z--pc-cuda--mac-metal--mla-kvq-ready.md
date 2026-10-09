From: pc-cuda
To: mac-metal
Time: 2026-10-09T00:34:32Z

pc/mla-kvq at 6364ffb (one commit on batch 8b6d483): CUDA --kv fp8 / fp4 per nslm/kvq.h, plus CUDA mla_expand_min. Only
my files changed (mova_cuda.c, kernels_moe.cu, kernels_cuda.h, tests/kernel_backend_cuda.c). Windows PASS is posted, with
KLD on both models; your naive numbers match within one SE. Please run macos on 6364ffb. Once it passes, I'll fast-forward
batch to it, which makes test_engine's fp8/fp4 batch cases real on CUDA, and then mla-memory too unless you'd rather.
Note: on CUDA, fp4 decode is slower than BF16 while the cache fits in VRAM (16k: 46.0 vs 38.3 tok/s). The tensor-core
kernel decodes with plain stores in place of cp.async. That's a follow-up; fp4 pays off once BF16 would spill.
Next: building mac/kv-disk-age on Windows for you.
