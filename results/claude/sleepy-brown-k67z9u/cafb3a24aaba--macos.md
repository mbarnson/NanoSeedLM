Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: cafb3a24aaba37e103c38050c212ecba87a198b2
Platform: macos
Agent: mac-metal
Time: 2026-11-04T23:49:42Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (rm -rf out): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 29 of 29 test programs, none skipped.
tools/test_nanoseedlm_k2.py (oMLX venv, OMLX_K2_MODEL set): PASS. The new nslm/ helpers (st_willneed, ns_row_prefetch,
ple_prefetch) build without warnings. The CUDA changes are not built on macOS.
