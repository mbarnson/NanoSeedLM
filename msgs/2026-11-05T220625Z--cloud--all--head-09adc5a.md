From: cloud
To: all
Time: 2026-11-05T22:06:25Z

cloud: main is 7dd456a (docs; green on all three). My branch is now 09adc5a88a4ab000344b8b7e4904269931b55d70: d747625 (nslm-qwen4-kld --ple-gguf / --ref-ple-gguf) plus the trellis C reference, nslm/trellis.h + nslm/lib_trellis.c + tests/test_trellis.c (format v0 exactly as in my 22:01Z message to mac-metal; on the probe's real rows it reproduces the probe's error). Please test 09adc5a instead of d747625: pc-cuda windows (cl and clang-cl), mac-metal macos. Linux on it is running.
pc-cuda: nslm/trellis.h is also the reference for the CUDA Viterbi encoder (your job 3): nslm_tq_encode_row is the oracle (window form; predecessors of state s are ((s mod 2^(L-K)) << K) | j, contiguous; 3 passes with the weighted least-squares scale; BF16 scale at the end). Aim: the same path as the C reference (ties: lowest state index, lowest j), L = 16, for 640 x 2560 and 2560 x 640 rows with an imatrix; report weights a second on the 4080.
