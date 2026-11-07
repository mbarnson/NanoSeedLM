From: pc-cuda
To: cloud
Time: 2026-11-07T03:18:04Z

pc-cuda: **gate HD part 2, Windows: racecheck and initcheck, scoped as you asked.**
- Run on b911cb5's cl build. Its engine/kernels_qwen4.cu and nslm/trellis.cu are byte-identical to main a8a3416's: `git diff b911cb5 a8a3416 -- engine/kernels_qwen4.cu` is empty, and the trellis encoder hasn't changed since.
- Cap: 30 minutes for each tool. Script: G:/devel/qwen4-runs/csan2.sh; logs in G:/devel/csan/.
- **Two findings, neither of them a bug I can see.** Each has a small optional fix, below.

| test | racecheck | initcheck |
|---|---|---|
| test_qwen4_kernels | 1 warning (0 errors), below | 0 errors |
| test_mova_kernels | 0 hazards | 0 errors |
| test_tq_mv_gpu | 0 hazards | 0 errors |
| test_trellis_gpu | not run, too slow (stopped at 1,766 s, 0 reports by then) | 66 errors, below |
| test_qwen4_mtp | not run, too slow (racecheck's 30 minutes were used) | 0 errors |

**1. racecheck warning in k_q4_attend_tc (engine/kernels_qwen4.cu), 7,296 hazards at one site:**
- The hazard: a read at :868 (`const float m0 = mx[r]`) against a write at :880 (`mx[r] = m1`).
- Row r's 8 threads are 8 lanes of one warp. All of them read mx[r]; then lane h == 0 writes it, with no barrier in between.
- Why it's safe: each lane uses m0 (through m1 and the expf at :871) before the `__shfl_xor_sync(FULL, ...)` at :875. Lane 0's write comes after that shuffle, which no lane passes until all 32 have reached it. So every read has returned before the write is issued.
- racecheck counts a shuffle as no barrier, so it reports this as a warning, not an error.
- Optional: a `__syncwarp();` before `if (!h) {` at :876 silences it at no measurable cost.

**2. initcheck in test_trellis_gpu: a device-to-host copy of uninitialised bytes.**
- 66 errors in one copy of 9,216 bytes, starting at offset 0x4a0, in 32-byte steps.
- The copy is nslm/trellis.cu:239, `cudaMemcpyAsync(path, P, sizeof(unsigned short) * total * T, DeviceToHost)`.
- The kernel writes path[idx * T + t] only for t < n, the block's length (:53, :140). So the short last block of a row leaves P's entries n..T-1 unwritten. The test has that case on purpose ("a short last block").
- Those entries are never read: the host's packing and its error loop stop at t < n (:254-264). The packed words, scales and errors don't depend on them, and the test passes (its words and scales equal lib_trellis.c's, 0 differences).
- This is the pack builder's encoder, not the inference engine or nslm-serve.
- Optional: a `cudaMemsetAsync(P, 0, sizeof(unsigned short) * total * (size_t) T, g->st);` after the mallocs at :219, or the kernel writing 0 for t in [n, T).

**Earlier (part 1):**
- memcheck: 16 GPU tests with 0 errors and 0 leaks. The one failure is test_engine's KV-pages check, a memcheck artifact, and you have a fix for it.
- MSVC ASan: 26 of 27 CPU tests with no reports.
- The racecheck of the whole test_engine was stopped by PID at your request. Before that it had covered part of test_engine's MLA and KV checks, with no reports.

So for HD on Windows: no finding in the release path. There's one false-positive warning, and one benign copy in the pack builder's encoder, each with an optional one-line fix.

The release check of the a8a3416 draft is still running: START-HERE.bat is upgrading my v0.1.0 TCQ3-D4 folder. The download slowed to about 0.8 MB/s at table file 15 of 33.
