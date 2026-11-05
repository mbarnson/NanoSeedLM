Status: PASS
Branch: pc/fp4-decode-gap
Commit: a085dce8072473b07a5026b47d2a46e9407e090d
Platform: macos
Agent: mac-metal
Time: 2026-11-05T02:48:11Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (rm -rf out): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 29 of 29 test programs, none skipped.
test_mova_kernels on Metal: the two new cases run as plain 40-split latent attention, both 0 mismatches
(3 rows x 32 heads r 768, 3 rows x 20 heads r 64); the FP8 and FP4 caches check 6 latent cases (4 before). The rest of
test_mova_kernels' output is the same as on main c404931. The CUDA changes are not built on macOS.
