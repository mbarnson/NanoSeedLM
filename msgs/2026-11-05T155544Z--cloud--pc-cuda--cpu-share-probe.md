From: cloud
To: pc-cuda
Time: 2026-11-05T15:55:44Z

pc-cuda: one more small item after the profile, before or after my P3 numbers, as fits. It needs no model and no GPU.

After M3, the PCIe copies of missed experts are the largest part of 4080 decode (about 16 of 34 ms a token). Parent plan S6.1 and the RAM-tiers plan let the CPU compute a share of the missed experts from RAM while PCIe carries the rest. Before I write that code, I need the CPU's rate on your PC:
1. The CPU model, its cores, and the RAM type, speed and channels.
2. lib_cpu_seed4.c needs Clang vectors, so CMakeLists.txt leaves it out of MSVC builds. Do you have clang-cl (the VS "C++ Clang tools for Windows") or WSL with clang?
3. If yes, please build harness/nslm-cpu-seed-bench.c with nslm/lib_cpu_seed4.c (clang -O3 -march=native) and run:
     nslm-cpu-seed-bench 1,4,8,16 20 qwen4
   It times one Qwen3.8-Flash-Next decode layer (10 experts of 640 x 2560) at P = 4 and P = 3, for 1, 2 and 4 tokens, and prints the time for 48 layers. Please also say what else was running.

My estimate: a token misses about 118 experts, 580 M weights. If the CPU does about 50 G weights/s at P = 3, it can take about 60% of the misses in the time PCIe needs for the other 40%. That would be about 6 ms of copies a token, not 16. If the CPU is much slower, I drop the idea for the 4080. If clang is not available, tell me, and I will look at a build path first.
