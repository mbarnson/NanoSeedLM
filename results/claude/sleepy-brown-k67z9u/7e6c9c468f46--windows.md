Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 7e6c9c468f4609ab3bb3df7f8286e5a4e5c286e2
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T03:23:44Z

win\build.bat test on 7e6c9c4 (still on main 350481b; MSVC 14.44, CUDA 13.1, RTX 4080), fresh out/, incremental build:
ctest 33/33 passed, incl. the new test_emulate and test_searchc (lib_emulate's pthreads / pwrite build with win/compat).
Skipped: test_affine, test_mova_cfg, test_tokenizer (no model dir).
