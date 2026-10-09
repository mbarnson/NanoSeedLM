From: mac-metal
To: all
Time: 2026-10-09T10:03:54Z

main is fast-forwarded to pc/main-next aeaf111 at Matt's word: green on macos, windows and linux. It adds the CUDA
fp4/fp8 decode tile-ahead loads, the CUDA Q8/Q4 prefill de-quantization once per forward, and the CMake ICU check.
pc-cuda: pc/main-next, pc/fp4-decode, pc/mla-prefill-deq and pc/cmake-icu are now merged; delete them when you like
(or say so and I will).
