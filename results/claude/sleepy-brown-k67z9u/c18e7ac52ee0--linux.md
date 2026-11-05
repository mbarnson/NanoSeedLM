Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: c18e7ac52ee052789d4dd2f3fb31ad3b49f21ba9
Platform: linux
Agent: cloud
Time: 2026-11-05T19:39:59Z

Linux, HF job 6aecdb14 (RTX PRO 6000 Blackwell, CUDA 12.8, Clang): ctest 40/40 PASS (3 skipped: test_affine, test_mova_cfg, test_tokenizer). The D4 check: the first decode step equals the GPU bit for bit, 1.1e-3 after 20 steps against 9.6e-2 with the tail dropped; a thrashing cache equals every expert in VRAM at all 20 steps. 111 warnings (108 -Wpsabi).
