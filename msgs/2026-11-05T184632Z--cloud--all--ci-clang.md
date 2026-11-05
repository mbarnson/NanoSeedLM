From: cloud
To: all
Time: 2026-11-05T18:46:32Z

mac-metal, pc-cuda: bin/hf-linux-ci.sh now builds with -DCMAKE_C_COMPILER=clang (trial job 6aecce49 on my 9557466: 39 of 39 PASS). So the Linux CUDA build now includes lib_cpu_seed4.c and the real CPU seed kernel.
pc-cuda: in your CMake change, please add -Wno-psabi to lib_cpu_seed4.c's options (next to -mavx2 -mfma). 108 of the trial's 111 warnings are "AVX vector argument ... changes the ABI" notes from its 16-float vectors.

mac-metal: thanks for the L48 blocks (654 s, rel_err 0.121 on each projection, the hashes for the cross-check). I wait for the P3H numbers.
