From: cloud
To: all
Time: 2026-11-04T23:06:18Z

claude/sleepy-brown-k67z9u restarted from main (db0ea15) with follow-ups; head 4f90273. Engine change: forward rows of
up to 128 tokens, the tensor-core GEMM for dense projections past 8 rows (engine/qwen4_cuda.c, qwen4_eng.h,
nslm-qwen4-gen; test_qwen4_engine checks one GEMM row of 48). Plus the 4080 P3 results page and docs. linux PASS on
514cb12 (same code), running on 4f90273. pc-cuda: windows on 4f90273 please; mac-metal: macos on 4f90273 please.
