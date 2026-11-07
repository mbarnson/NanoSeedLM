# Plan: an nslm-serve that a stranger cannot use (2026-11-06)

**Goal.** A user can run `nslm-serve` as a server, and another person cannot easily take advantage of it. That person
can be a web page in the user's browser, a machine on the same network, or a crafted request or model file. Matt's
words: "if someone sets this up as a server they aren't easily taken advantage of."

**Status.** Done, for v0.1.1. H1 to H3 are done, and HA to HE pass. H6 and H7 are for v0.1.2.
- HA: test_serve_guard passes on Linux, macOS and Windows, at 8bfed94 and f6d5402.
- HB: the first run failed 7 tests. The fixes:
  - the safetensors header and shape bounds;
  - JSON literals read past the buffer (mac-metal);
  - aligned test fixtures;
  - leaks on the skip paths.
  The job passes from 6e13dd6.
- HC: Windows passes (`dumpbin`: Control Flow Guard in the cl and clang-cl builds of nslm-serve.exe). Linux passes
  (HF job `6aee90d9`, GCC, Release, a8a3416): nslm-serve and nslm-chat are PIE, with full RELRO (GNU_RELRO and
  BIND_NOW), `__stack_chk_fail`, FORTIFY's `_chk` functions (14 and 12) and a stack that is not executable.
- HD: memcheck finds nothing in 17 GPU tests, and MSVC ASan nothing in 26 CPU tests. mac-metal's Metal validation
  finds nothing once a test binds its buffers (b273911). On Windows, pc-cuda ran racecheck and initcheck for up to 30
  minutes each, on b911cb5's cl build (the same kernels as a8a3416). They find nothing in test_mova_kernels and
  test_tq_mv_gpu, and initcheck finds nothing in test_qwen4_mtp. They find two items, and both are explained:
  - racecheck, test_qwen4_kernels: 1 warning (0 errors) in `k_q4_attend_tc`. The 8 lanes of row r read `mx[r]`, and
    then lane 0 writes it. Each lane uses its read before the `__shfl_xor_sync`. Lane 0 writes after it, so the write
    cannot come first. The tool does not count a shuffle as a barrier. v0.1.2 can add a `__syncwarp()`.
  - initcheck, test_trellis_gpu: 66 errors in one copy to the host (`nslm/trellis.cu`, the path buffer). The short last
    block of a row writes entries 0 to n-1 of its path, and the copy takes all T entries. The host reads only entries
    under n, and the test's words and scales equal lib_trellis.c's. This is the pack builder's encoder, not the
    engine. v0.1.2 can clear the buffer.
  - racecheck did not finish 3 tests. It stopped test_trellis_gpu at 1,766 s and test_engine after 3½ hours, both
    with no reports. It had no time left for test_qwen4_mtp.
- HE: passes in all three modes, on macOS and Windows. `--host 0.0.0.0` without a key refuses to start. The refusal
  comes after the model load, which v0.1.2 can move before it.

## Why

pc-cuda's review of `harness/nslm-serve.c` (`msgs/2026-11-06T232556Z--pc-cuda--cloud--security-tooling-v011.md`):

- The server binds 127.0.0.1 by default, caps a body at 16 MB, and times out slow clients. That part holds.
- Each reply had `Access-Control-Allow-Origin: *`, and the server had no key. So any web page in the user's browser
  could send a prompt to `http://127.0.0.1:8080` and read the answer.
- The server did not check the Host header. So a page could reach it through DNS rebinding as well.
- No sanitizer or static check ran in CI on the C code that reads requests and model files.

## Steps and gates

| Step | Work | Who | Gate |
|---|---|---|---|
| H1 | `harness/serve_guard.h`: an API key (`NSLM_API_KEY`), needed off loopback. Without a key, only a loopback Host. A browser `Origin` only with `--allow-origin`. No wildcard CORS | cloud | HA, HE |
| H2 | `-DNSLM_SANITIZE=ON` (ASan, UBSan, LSan) and `.github/workflows/sanitize.yml`, which runs the tests with it on each push | cloud | HB |
| H3 | Hardening flags in each build: `-fstack-protector-strong`, `_FORTIFY_SOURCE=2`, full RELRO, Control Flow Guard (MSVC) | cloud | HC |
| H4 | compute-sanitizer (memcheck, racecheck, initcheck) over the GPU tests, and an MSVC `/fsanitize=address` build of the CPU tests | pc-cuda | HD |
| H5 | `tools/serve_guard_check.sh` against a running server of the release candidate: no key, a key, `--allow-origin` | pc-cuda, mac-metal | HE |
| H6 | Fuzz targets (libFuzzer with ASan) for the HTTP reader, the JSON parser, the chat template and the model file readers | cloud | v0.1.2 |
| H7 | CodeQL code scanning on each push to `main` | cloud | v0.1.2 |

- **HA.** `test_serve_guard` passes on the three platforms. It covers the loopback names, foreign Hosts, keys that
  differ by one byte, and origins.
- **HB.** The sanitizer job passes on the release candidate, or each finding is fixed first.
- **HC.** The release builds have the flags (`checksec` on Linux, `dumpbin /headers` on Windows), and the tests pass.
- **HD.** No finding, or each finding is fixed or explained before the release.
- **HE.** `serve_guard_check.sh` passes in each mode, on Windows and macOS. `--host 0.0.0.0` without a key does not
  start.

## Costs

- Clients on the same machine do not change: the start scripts serve on 127.0.0.1 without a key.
- A web UI in a browser needs `--allow-origin` now. A server on a LAN address needs a key.
- The sanitizer job uses GitHub's free Linux runners. No HF jobs.
