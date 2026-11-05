Status: FAIL
Branch: claude/sleepy-brown-k67z9u
Commit: 39967adb8a40dcbd53085543b966b2f467fba596
Platform: macos
Agent: mac-metal
Time: 2026-11-05T06:05:26Z

macOS 27.2, M4 Max 128 GB, AC power. Clean build (fresh worktree): PASS, with the one old warning (nslm-mova-score.c:157).
make test with MOVA_DIR = the K2-Horizon-MoVA snapshot: all tests passed, 31 of 31 test programs, none skipped;
test_qwen4_kernels on Metal: 14 of 14.
tools/test_nanoseedlm_k2.py (oMLX venv, OMLX_K2_MODEL set): FAIL in test_load_mla, _load_mla line 440:
"assert err < 0.02" got 0.2334 for the 40-token ids. It passes on main 047ad9f.
Cause (bisected): MATVEC_MAX_PAIRS 32 -> 64 alone (the broadcast change is fine: limit 32 with it passes, and
test_switch_linear passes). At 32, SeedLinear's 40 prompt rows take the seed GEMM, and the model's logits equal the BF16
reference exactly (err 0.0). At 64 they take the mat-vec. That is right too (about 2e-3 to 5e-3 from the GEMM per call;
gather_matvec and gather_gemm agree with ref_decode to 2e-3 for P3, P4 and P8 at 1 to 65 rows), but it flips near-tie
top-k routing at three of 40 positions (errors 0.42, 0.17, 1.31; the rest 0.01). No other module changes path
(SeedHeads stays on the GEMM at 4 x 40 rows).
