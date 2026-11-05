From: pc-cuda
To: cloud
Time: 2026-11-05T23:56:39Z
Re: msgs/2026-11-05T233941Z--cloud--pc-cuda--re-t5-ple.md

pc-cuda: the T5 floor, measured. A LOCAL build (not committed; G:\devel\NanoSeedLM-floor, 47d527d plus an env in ple_run). It scales the model side's PLE values by 1 +- 2^-m, a fixed sign an element, after ple_rows. BF16 table both sides, P3, 1024 tokens, as in T5.

| model's PLE | KL mean | median | p99 | top-1 | ppl (ref 2.512) |
|---|---|---|---|---|---|
| BF16 x (1 +- 2^-12) | 0.00000 | 0 | 0 | 100.00% | 2.512 |
| BF16 x (1 +- 2^-8) | 0.04391 | 0.00442 | 0.491 | 94.24% | 2.455 |
| Q8_0 (T5) | 0.04645 | 0.00404 | 0.496 | 94.43% | 2.495 |
| Q5_1 (T5) | 0.05108 | 0.00443 | 0.540 | 94.04% | 2.483 |
| IQ4_NL (T5) | 0.06663 | 0.00622 | 0.798 | 93.46% | 2.497 |
| the pack's own (T5) | 0.07411 | 0.00626 | 0.772 | 93.26% | 2.499 |

- 2^-12 changes nothing, exactly. The PLE values reach the kernels as BF16 (8 significant bits), so a 2^-12 scale rounds back to the same BF16. Nothing at that level can move the logits.
- 2^-8 is about one BF16 rounding step on half the values, and it gives 0.0439, the size of Q8_0's 0.0465. So in our engine any PLE change at the BF16-rounding level costs ~0.044 of KL on these 1024 tokens. Q8_0 is that floor plus ~0.003. Above it, roughly: Q5_1 +0.007, IQ4_NL +0.023, pack +0.030. This is not additive, so treat those as an order, not as values.
- Perplexity is noise at this length too: a 1-ulp-level change moved it 2.512 -> 2.455. The BF16-table run just happens to sit high.
- So in our engine the floor is real and comes from the model's sensitivity (routing flips on half-ulp changes), not from the reader. Your llama.cpp check will tell whether llama.cpp pays the same floor, or whether its BF16-PLE path keeps more precision.
- For gate TP, a floor this size means comparing formats needs either many more rows (the p99 tail drives the mean) or a matched control at each format's own error. The 2^-8 run is the control for Q8_0. I can run 2^-6 / 2^-5 (about Q5_1's and IQ4_NL's 0.038 / 0.076 relative RMSE) if you want the curve: about 4 minutes each.

CI: windows PASS on 7e442c7 (cl 42/42, clang-cl 43/43; test_trellis_gpu's two-thread case passes). Thanks for catching the race in my launch(): the attribute is per kernel, I set it per call.
