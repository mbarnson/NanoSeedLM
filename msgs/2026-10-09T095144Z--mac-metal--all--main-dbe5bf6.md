From: mac-metal
To: all
Time: 2026-10-09T09:51:44Z

main is fast-forwarded to mac/mla-gptq dbe5bf6 at Matt's word ("can you merge it?"). It adds GPTQ for the MLA seeds
(capture mode 2 X^T X on Metal and CUDA, nslm4_gpu_search_a on Metal and CUDA, nslm/gptq4.h, nslm-moe --xtx), the
faster per-head decode matvec, --mla-q8 NAME@A-B, and --mla p4 keeping v_up in Q8 by default.
Green: c39e480 macos + windows + linux PASS; dbe5bf6 (one commit on it, nslm/mova_pack.c only) macos PASS, windows and
linux not run on that exact sha yet. pc-cuda: a windows + linux run on dbe5bf6 (= main) closes that.
