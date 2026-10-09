Status: PASS
Branch: mac/kv-pages
Commit: 25a0b304dc2feeb6e7d8560d097e1a2ccd421a36
Platform: windows
Agent: pc-cuda
Time: 2026-10-09T01:26:45Z

Windows 11, MSVC + CUDA 13.1, RTX 4080: ctest 18/18 passed (test_affine skipped). test_engine prints "MLA KV pages: not paged in this engine", as intended.
