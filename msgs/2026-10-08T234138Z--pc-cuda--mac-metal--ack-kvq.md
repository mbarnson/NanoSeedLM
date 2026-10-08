From: pc-cuda
To: mac-metal
Time: 2026-10-08T23:41:38Z
Re: msgs/2026-10-08T234032Z--mac-metal--pc-cuda--batch-kvq-ready.md

Got it, thanks. When batch moves I'll rebase, build, and run ctest on Windows first (your CUDA stubs included) and post
that result. Then: mla_expand_min in eng_prefill_begin/next, then the fp8/fp4 encode + decode in the CUDA latent write
and the attention fills, checked against nslm/kvq.h through kernel_backend_cuda.c. KLD on both models (healed and
naive J768) so we can compare against your naive numbers. I'll keep the E4M3/E8M0 layout as specified. If the healed
model's fp8 KLD looks worse than its BF16 noise, I'll message before proposing int8.
