Status: PASS
Branch: mac/tq-head-test
Commit: a611a5ef580fe507931922227f9cff33dcf5931d
Platform: macos
Agent: mac-metal
Time: 2026-11-06T04:53:06Z

macOS, M4 Max, clean worktree at a611a5ef580fe507931922227f9cff33dcf5931d (mac/tq-head-test on 7e42e52: test_qwen4_metal's trellis-head check): make build, then make test with MOVA_DIR. 38 of 38 pass, 0 skipped; test_qwen4_metal's new tcq_head_check passes (the head opens with a bank L4; 1 to 3 drafts give the 24 tokens of decode without drafts; draft 1 511 with the head bank against 171 without it, as the C reference). test_nanoseedlm_k2 PASS. Warnings: harness/nslm-mova-score.c:157 and tests/test_chat_template.c:56 (not mine).
