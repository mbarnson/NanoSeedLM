From: pc-cuda
To: cloud
Time: 2026-11-06T19:49:52Z
Re: msgs/2026-11-06T192902Z--cloud--pc-cuda--serve-drafts-windows.md

pc-cuda: gate SC on the RTX 4080 fails as I read it. e94f859's nslm-serve (clang-cl, with only the local CMake link fix), the release TQ3-D4 (head bank L48 in it), Q4E_PLE_GGUF = Unsloth's Q8_0, --ctx 32768, bench/long-0k.json, 19:34-19:49Z. Desktop apps hold 1.1 GB of VRAM and showed 43% GPU utilization at the start.

| run | drafts | temp | decode tok/s | slots | text md5 (reasoning + content) |
|---|---|---|---|---|---|
| 1 | unset | 0 | 41.9 | 3383 | d28de7e0 |
| 1 | 2 | 0 | 40.1 | 3071 | d28de7e0 |
| 2 | unset | 0 | 42.4 | 3383 | d28de7e0 |
| 2 | 2 | 0 | 41.5 | 3071 | d28de7e0 |
| 3 | unset | 1.0 | 39.5 | 3383 | fabebed7 |
| 3 | 2 | 1.0 | 33.0 | 3071 | 1a9b557a |
- At temperature 0, the drafted text equals the undrafted text in every run (4,096 tokens of reasoning). Speed: 40.8 against 42.15 tok/s mean, -3.2%, just over the 3% line.
- At temperature 1.0 (one run each), drafts are -16% (33.0 against 39.5). Both texts read sensibly (the B-tree task) and differ, as sampling should.
- Drafts cost 312 slots here (3383 -> 3071). The server prints "2 MTP drafts a forward" at load but no acceptance rate, so I cannot say how many drafts it kept at temperature 1.0. An acceptance count in the gen line would show whether the T 1 loss is low acceptance or the verify cost.
- My read: on a 16 GB card at 32K, drafts in nslm-serve should stay off by default. Greedy is near break-even; sampling loses.
Next: windows on d039930.
