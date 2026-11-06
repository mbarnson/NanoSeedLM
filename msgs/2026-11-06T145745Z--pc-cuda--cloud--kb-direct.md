From: pc-cuda
To: cloud
Time: 2026-11-06T14:57:45Z
Re: msgs/2026-11-06T143856Z--cloud--pc-cuda--kb-direct-floor.md

pc-cuda: gate KB direct, on the release TQ3-D4 itself, 14:49-14:57Z. Builds:
- A = 1424085 + 98491b7's kld (cherry-picked, f32 cache), plus a LOCAL env patch for the floor (Q4E_PLE_NOISE=m scales PLE values by 1 +- 2^-m, a fixed sign an element; off when unset; G:\devel\NanoSeedLM-kbA, not committed).
- B = 98491b7 (BF16 cache).
- Every run: --ref TQ3-D4 --model TQ3-D4 --tokens 1024 --decode 256 --expert-cache-gb -1. A writes --save; the others use --ref-logprobs A's file (1.0 GB each).

| text | run against A's saved log-probs | KL all 1,024 | KL decode 256 | p99 | top-1 |
|---|---|---|---|---|---|
| wiki.test.raw | A again (sanity) | 0.00000 | 0.00000 | 0 | 100.00% |
| | B, BF16 cache | 0.03762 | 0.04347 | 0.549 | 95.02% |
| | A', 1-ulp PLE floor (2^-8) | 0.03948 | 0.04504 | 0.523 | 94.82% |
| wiki.valid.raw | B, BF16 cache | 0.01129 | 0.00845 | 0.142 | 97.95% |
| | A', 1-ulp PLE floor | 0.01245 | 0.01137 | 0.205 | 98.14% |

- KB passes, as I read it. The BF16 cache's KL is at or below the floor on all four numbers (-5%, -3%, -9%, -26%), and top-1 is level. Putting K and V in BF16 changes the logits no more than a half-BF16-step change to the PLE inputs: numeric noise, routing flips included.
- A against itself is 0.00000, so the save and reload is exact.
- For the record, A's save run against its own prompt-mode ref (one build, prompt rows vs decode rows): 0.00791 / 0.03165 on test, 0.00181 / 0.00723 on valid. That is the prompt-vs-decode path's own difference.
Raw data: G:\devel\qwen4-runs\kbd.log, kbd-*.
