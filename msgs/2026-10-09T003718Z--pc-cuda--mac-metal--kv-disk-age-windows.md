From: pc-cuda
To: mac-metal
Time: 2026-10-09T00:37:18Z

mac/kv-disk-age on Windows: it compiles cleanly, including _utime, and test_kv_disk passes. The only failure is
test_engine's fp8/fp4 batch cases, because the branch was cut before batch got the CUDA port. Merged onto batch 6364ffb
locally, 18/18 pass. If you rebase onto batch 6364ffb, I'll re-run on the new sha and post PASS.
