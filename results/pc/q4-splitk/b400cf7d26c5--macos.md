Status: PASS
Branch: pc/q4-splitk
Commit: b400cf7d26c5439718c6d29bbaf03eec54d112a3
Platform: macos
Agent: mac-metal
Time: 2026-11-05T15:17:35Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (fresh worktree): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 34 of 34 test programs, none skipped.
test_mova_kernels' new token-0 check (T = 1 against T = 2..8, bit for bit) passes on Metal for bf16, seed4, q8, q4,
seed4p4, seed6p8 at K = 256, 512 and 4096 (no FAIL lines). tools/test_nanoseedlm_k2.py: PASS.
