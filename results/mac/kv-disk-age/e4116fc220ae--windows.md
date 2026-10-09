Status: FAIL
Branch: mac/kv-disk-age
Commit: e4116fc220aed3e43f77b3d3a213d9b86fa0f10c
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T00:37:15Z

Windows 11, MSVC + CUDA 13.1, RTX 4080, clean build: compiles with no warnings (the Windows plat_set_mtime / _utime path
builds). test_kv_disk PASS. ctest 17/18: test_engine fails only in "MLA fp8 batch" / "MLA fp4 batch" (eng_open: KV format
not supported by the CUDA engine), because this branch predates the CUDA fp8/fp4 port. Not caused by this branch.
Merged locally onto batch 6364ffb (6f4a179, not pushed): ctest 18/18 PASS. Rebase onto batch and it should be green on
windows.
