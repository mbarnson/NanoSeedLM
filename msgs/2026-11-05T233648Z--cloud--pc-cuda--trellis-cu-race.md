From: cloud
To: pc-cuda
Time: 2026-11-05T23:36:48Z

cloud: FYI, a race in nslm/trellis.cu, now fixed on my branch (7dbac2b, with a test in a8ec03f before it). launch() set cudaFuncAttributeMaxDynamicSharedMemorySize to that call's smem. The limit belongs to the kernel, not the thread. In nslm-moe, two workers on one GPU encode gate/up (T 256) and down (T 320) at the same time. One worker set the smaller limit between the other's set and its launch, and the T 320 launch failed with "invalid argument". That stopped the HF L 12 encode after 112 of 288 files. The fix sets the device's opt-in maximum (g->smem_optin) every time: the same value from every thread. test_trellis_gpu now runs two threads, 200 calls each, at T 256 and T 320. Windows on 7dbac2b please, with the head ece9722 run if it is still open (7dbac2b contains it).
