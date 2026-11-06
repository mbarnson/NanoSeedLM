Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 8856cc47e8ed793ad3fb4017451d6a38d71b3bf9
Platform: macos
Agent: mac-metal
Time: 2026-11-06T10:02:39Z

macOS, M4 Max: clean worktree at 8856cc47e8ed793ad3fb4017451d6a38d71b3bf9 (claude/sleepy-brown-k67z9u: nslm-qwen4-emulate --dense-only, hf_llama_kld.sh dense:KIND; hf_release_publish.sh copies to local disk).
- make build, then make test with MOVA_DIR: 38 of 38 pass, 0 skipped ("all tests passed"). test_emulate, test_moe, test_qwen4_metal and test_trellis_gpu PASS.
- test_qwen4_metal's trellis head check: unchanged (bonus 80, drafts 511 299 9; without the head bank 171 341 317).
- test_nanoseedlm_k2 (oMLX venv Python) PASS.
- Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
