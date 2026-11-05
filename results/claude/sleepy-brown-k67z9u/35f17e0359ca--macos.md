Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 35f17e0359caae9f9b3d384a18521c8fb4986a33
Platform: macos
Agent: mac-metal
Time: 2026-11-05T16:19:33Z

macOS, M4 Max, clean worktree at 35f17e0359caae9f9b3d384a18521c8fb4986a33: make build, then make test with MOVA_DIR. 34 of 34 pass, 0 skipped (test_qwen4_kernels, test_qwen4_ref and test_qwen4_metal PASS). test_nanoseedlm_k2 PASS. The qwen4_api.c warning is gone; only harness/nslm-mova-score.c:157 is left.
