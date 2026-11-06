Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: da4e1cebd04c7927b1f376f69c19cf34c4f675f6
Platform: windows
Agent: pc-cuda
Time: 2026-11-06T20:49:32Z

Windows 11, RTX 4080, CUDA 13.1, Ninja Release, G:\devel\NanoSeedLM-lfsr at da4e1ce (clean tree, reconfigured; A1 + A2).
- cl (MSVC 14.44): ctest 45/45 PASS (3 skipped: test_affine, test_mova_cfg, test_tokenizer).
- clang-cl: ctest 46/46 PASS (same 3 skipped).
Draft lines (cl build):
test_qwen4_mtp: "prompt of 16: draft probabilities 0.0119 0.0075 0.0087 (the reference's 0.0119 0.0075 0.0087)"; "prompt of 16: bonus 76, drafts 217 194 145 (the C reference's)"; "prompt of 45: draft probabilities 0.0075 0.0080 0.0070 (the reference's ...)"; "prompt of 45: bonus 177, drafts 122 250 429 (the C reference's)"; "drafts through a cache of 8 slots: 170 misses"; "greedy decode of 40 tokens with 1 / 2 / 3 drafts a step: 0 accepted, the first 40 tokens as without".
test_qwen4_pick: "Q4E_DRAFT_MIN_P 0.5: 0 of 0 drafts kept in 40 forwards"; "Q4E_DRAFT_MIN_P 2: 0 of 0 drafts kept in 40 forwards"; "sampled drafts at temperature 1.0: 0 of 114 kept".
(The tiny fixture's head accepts no drafts, so these show the paths run and the text is unchanged, not acceptance. Its draft probabilities are ~0.01, so a 0.5 limit verifies none, hence "0 of 0".)
Log: G:\devel\build-da4e1cebd04c.log.
