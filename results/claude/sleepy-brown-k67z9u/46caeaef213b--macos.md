Status: FAIL
Branch: claude/sleepy-brown-k67z9u
Commit: 46caeaef213b85161c0dc1f114e17749d8e7536e
Platform: macos
Agent: mac-metal
Time: 2026-11-05T18:11:34Z

macOS, M4 Max, clean worktree at 46caeaef213b85161c0dc1f114e17749d8e7536e: make build OK; make test 34 of 35 pass, 0 skipped. test_nanoseedlm_k2 PASS.

FAIL: test_qwen4_mtp, at tests/test_qwen4_mtp.c:134, "the cache never missed". It printed "drafts through a cache of 0 slots: 0 misses". The Metal engine has no expert cache (q4e_cache_stats gives 0 slots and 0 misses), so m > 0 cannot hold there. The drafts-equal half of the same check passed. Suggested fix in your file: CHECK(slots == 0 || m > 0, ...), i.e. skip the miss count when the engine reports no cache.

New warnings on this SHA:
- tests/test_qwen4_metal.c:35, missing field 'tail_skip' initializer (my file; I fix it on my branch with a memset, as test_qwen4_mtp does).
- tests/test_chat_template.c:56, missing field 'no_thinking'.
