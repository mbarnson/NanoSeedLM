Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 14daf4a7b45af54c1bc4b4550a6b806701dbef68
Platform: macos
Agent: mac-metal
Time: 2026-11-06T00:43:12Z

macOS, M4 Max, clean worktree at 14daf4a7b45af54c1bc4b4550a6b806701dbef68 (claude/sleepy-brown-k67z9u on main 7e442c7: trellis experts in the CUDA engine, MF_TCQ in kernels_moe.metal's enum, trellis_none.o in SEARCH_O): make build, then make test with MOVA_DIR. 37 of 37 pass, 0 skipped (test_trellis, test_engine, test_qwen4_kernels with 'tcq: not on Metal', test_qwen4_metal, test_qwen4_mtp, test_qwen4_api_mtp PASS). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine). The run shared the Mac with the K = 3 CPU encode.
