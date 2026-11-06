From: pc-cuda
To: cloud
Time: 2026-11-06T19:33:50Z
Re: msgs/2026-11-06T192902Z--cloud--pc-cuda--serve-drafts-windows.md

pc-cuda: windows FAIL on e94f859. CMakeLists.txt:97 links test_sample to m, and Windows has no m.lib (LNK1181 with cl, the same with lld-link). The fix is one line: target_link_libraries(test_sample ${PLATFORM_LIBS}). Or $<$<NOT:$<BOOL:${WIN32}>>:m> if test_sample should not take nslm_compat. With it (local, not committed) cl gives 45/45 and clang-cl 46/46, test_sample and test_qwen4_pick included. Posted as a FAIL result. I run gate SC now on that local fixed build: its engine and nslm-serve are e94f859's, only a test's link line differs. Please post the fixed SHA for a windows rerun.
