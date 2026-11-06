Status: FAIL
Branch: claude/sleepy-brown-k67z9u
Commit: e94f85994f3fea8e7e7933e95b1ba35c838a671a
Platform: windows
Agent: pc-cuda
Time: 2026-11-06T19:33:47Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release, G:\devel\NanoSeedLM-lfsr at e94f859 (clean tree, reconfigured).
FAIL: both builds stop linking test_sample.
- cl: "LINK : fatal error LNK1181: cannot open input file 'm.lib'"
- clang-cl: "lld-link: error: could not open 'm.lib': no such file or directory"
Cause: CMakeLists.txt:97 "target_link_libraries(test_sample m)". There is no libm on Windows. PLATFORM_LIBS (line 47/50) already carries m on Linux only.
With that one line changed locally to "target_link_libraries(test_sample ${PLATFORM_LIBS})" (not committed), both builds pass: cl ctest 45/45, clang-cl 46/46, test_sample 3.3 s, test_qwen4_pick 0.4 s (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
Logs: G:\devel\build-e94f85994f3f.log (as is), G:\devel\build-e94f859-fix.log (the fix).
