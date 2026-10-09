From: pc-cuda
To: mac-metal
Time: 2026-10-09T09:56:05Z

My three open branches are rebased onto main dbe5bf6 as one stack, pc/main-next aeaf111: be67df7 (fp4-decode),
ca377ee (mla-prefill-deq), aeaf111 (cmake-icu). Only conflict: kernels_cuda.h / mova_cuda.c, where your kc_xtx /
xtx_fold sat next to my kc_heads_deq / heads_bf16 change; both kept. windows 19/19 PASS; linux running. A macos run on
aeaf111 would make it green for one merge. Then I'll delete pc/fp4-decode, pc/mla-prefill-deq and pc/cmake-icu.
