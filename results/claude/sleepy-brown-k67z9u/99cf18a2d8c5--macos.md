Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: 99cf18a2d8c56fb93736ffe07c3ab96b9b4bab33
Platform: macos
Agent: mac-metal
Time: 2026-11-06T05:45:23Z

macOS, M4 Max: clean worktree at 99cf18a2d8c56fb93736ffe07c3ab96b9b4bab33 (claude/sleepy-brown-k67z9u: nslm-qwen4-pack --tq; ns_add_banks takes the head's bank by the routers).
- make build, then make test with MOVA_DIR: 38 of 38 pass, 0 skipped. test_emulate, test_engine, test_qwen4_kernels, test_qwen4_metal, test_qwen4_api_mtp and test_trellis_gpu all PASS.
- test_qwen4_metal's head check passes with the router rule. Drafts at 1–3 a step give the same 24 tokens as without drafts. "trellis head: bonus 80, drafts 511 299 9; without the head bank: bonus 80, drafts 171 341 317", the same as before the change.
- test_nanoseedlm_k2 (tools/test_nanoseedlm_k2.py, oMLX venv Python) PASS. `make test-mlx` with the default PY=python3 lacks mlx_lm.models.activations; that is my environment, not the commit.
- Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
