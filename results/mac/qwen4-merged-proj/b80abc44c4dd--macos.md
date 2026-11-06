Status: PASS
Branch: mac/qwen4-merged-proj
Commit: b80abc44c4dd4ed024e4b127f6986235cd78da64
Platform: macos
Agent: mac-metal
Time: 2026-11-06T18:08:10Z

macOS, M4 Max: clean worktree at b80abc44c4dd4ed024e4b127f6986235cd78da64 (mac/qwen4-merged-proj, from claude/sleepy-brown-k67z9u 4169259: merged input projections and hyper-connection rows on Metal).
- make build, then make test with MOVA_DIR: 38 of 38 pass, 0 skipped ("all tests passed").
- test_qwen4_kernels, test_qwen4_metal, test_qwen4_mtp and test_qwen4_api_mtp PASS with the same numbers as the base.
- test_nanoseedlm_k2 PASS.
- Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
