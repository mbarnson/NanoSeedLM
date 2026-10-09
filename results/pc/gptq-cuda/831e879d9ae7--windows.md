Status: PASS
Branch: pc/gptq-cuda
Commit: 831e879d9ae78608d60950dac9745f40dc20ef45
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T05:30:18Z

Windows 11, MSVC + CUDA 13.1, RTX 4080 (fresh CMake configure): ctest 19/19 passed (test_affine skipped; test_linalg newly registered in CMake), no C warnings. test_search4_gpu: "search4 full A GPU vs C: 0 blocks differ, 0 error bits differ, of 2400"; "gptq4: GPU vs C 0 blocks differ".
