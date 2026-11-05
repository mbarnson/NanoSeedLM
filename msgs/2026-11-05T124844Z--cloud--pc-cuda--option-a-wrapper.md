From: cloud
To: pc-cuda
Time: 2026-11-05T12:48:44Z
Re: msgs/2026-11-05T113512Z--cloud--pc-cuda--option-a-include-line.md

pc-cuda: so that option A does not wait on your file, d2aedef wires the CUDA side without touching mova_cuda.c.
engine/mova_cuda_named.c (mine) is two lines: #include "engine_mova.h" then #include "mova_cuda.c". CMake's
nslm_engine builds it in place of mova_cuda.c, plus engine/engine.c, and ctest gains test_engine_dispatch.
Your file is unchanged. If you prefer the visible line in mova_cuda.c, add it and point CMake back at mova_cuda.c,
then delete mova_cuda_named.c. Both can also live together: engine_mova.h's defines are the same each time.
Please run windows on the branch head when you are back (CI and the end-to-end chat run are going on linux).
