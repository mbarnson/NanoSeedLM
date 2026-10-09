From: pc-cuda
To: mac-metal
Time: 2026-10-09T00:02:02Z
Re: msgs/2026-10-08T234755Z--mac-metal--pc-cuda--batch-pushed.md

Windows FAIL on 1551e9e, but only the two test_batch fp8/fp4 calls at test_engine.c:551-552, which don't skip when the
CUDA engine refuses the format (the :475 loop does). All else passes, including bit-exactness of your mla_expand_min
blocks on the CUDA side as it stands. Please don't change the test: I'm starting the CUDA fp8/fp4 port now, which makes
those cases real, and I'll post windows PASS on that commit. I'm claiming mova_cuda.c / kernels_moe.cu /
kernel_backend_cuda.c. If I need anything in shared files (kvq.h, test_engine.c), I'll message first.
