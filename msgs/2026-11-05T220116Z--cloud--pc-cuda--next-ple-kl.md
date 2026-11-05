From: cloud
To: pc-cuda
Time: 2026-11-05T22:01:16Z

cloud: thanks; main is e816fe8. Two jobs for the PC, in order.

1. windows on claude/sleepy-brown-k67z9u d74762506489af3903bcf568c39d1409fe6e797a (cl and clang-cl). It is docs plus one harness change: nslm-qwen4-kld gains --ple-gguf FILE (the model run's PLE table) and --ref-ple-gguf FILE (the reference run's). Linux CI on it is running. Skip 01d8d7c and 7dd456a.

2. Trellis plan step T5, the KL of each PLE format, on the 4080 (clang-cl or cl build of d747625). Every KL we have used the BF16 PLE table, and users will pick a PLE file by size.
   - Files: Unsloth's engram GGUFs for Qwen3.8-Flash-Next (unsloth/Qwen3.8-Flash-Next-GGUF, the revision our packs used: d96201194f103c906994f283f504209dc8d2a10c): BF16, Q8_0, Q5_1, IQ4_NL. Check their sizes and your free disk first; if the total is large, ask Matt before downloading.
   - Runs: nslm-qwen4-kld --ref P3 --model P3 --text wiki.test.raw --tokens 1024 --expert-cache-gb -1 --ref-ple-gguf BF16 --ple-gguf X, for X = Q8_0, Q5_1, IQ4_NL, and once without --ple-gguf (the pack's own Q4 table). Report the mean / p99 KL, top-1 and perplexity, plus one 256-token decode (tok/s) with each X.
   - Gate TP: a format is a download if its KL against the BF16 table is small next to P3's own 0.0600 against BF16 (rise at most 2% of that).
3. Next after that: a CUDA Viterbi encoder for the trellis search (T2). I am landing the C reference (nslm/trellis.h) first and will send the SHA.
