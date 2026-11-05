Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: ecd552ff3926a8dbdcff6d349481d42b0e759b64
Platform: macos
Agent: mac-metal
Time: 2026-11-05T15:33:31Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (fresh worktree): PASS, with the old warning (nslm-mova-score.c:157) and
one new one: engine/qwen4_api.c:56:107: missing field 'mtp' initializer [-Wmissing-field-initializers] (the Q4EOpts
initializer). make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 34 of 34 test programs, none
skipped; test_qwen4_kernels on Metal 18 of 18 (norm_long_case included); test_qwen4_ref PASS. tools/test_nanoseedlm_k2.py:
PASS. nslm-qwen4-gen --drafts 2 on P3 (Metal): "this engine has no MTP drafts" after the first token, exit 1, as intended.
