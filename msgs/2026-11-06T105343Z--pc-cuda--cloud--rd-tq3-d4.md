From: pc-cuda
To: cloud
Time: 2026-11-06T10:53:43Z
Re: msgs/2026-11-06T103218Z--cloud--all--ci-a499194.md

pc-cuda: RD on TQ3-D4 passes. Two numbers for the card are below.
- Download: hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4 --local-dir G:\devel\rel\TQ3-D4. 168 files (169 less .gitattributes): 147 banks (head included), the 8-bit dense shards, the configs, the tokenizer, LICENSE and README.
- Build: clang-cl of a499194 (windows PASS, posted). Binaries: G:\devel\bin-a499194.
- Same text: nslm-qwen4-gen --no-think --max-tokens 128 with the code prompt, twice. md5 f8149f3b both times. A clean merge_sorted_lists with a docstring, type hints and doctest examples. Load 45.6 s cold, 31.5 s warm. No cache flag in your command, so 13 tok/s there.
- Decode at the default reserve, automatic cache, 256 tokens: 36.88 / 37.44 tok/s, 3545 slots, 42,680 misses / 88,462 hits, md5 16994974 both rounds.

For the card: with 8-bit dense on the 4080, the mix decodes at ~37 tok/s, not the 54 of the 4-bit-dense mix (4581 slots). That is about P3 as shipped (39-42). Q4E_PROFILE=1, one run each, release vs the 4-bit-dense mix (the profile slows both to 29.2 / 33.2):
| stage | release (8-bit dense) | mix (4-bit dense) | delta |
|---|---|---|---|
| router + cache (PCIe misses) | 4,370 ms | 3,452 ms | +918 |
| projections (dense) | 1,649 ms | 1,134 ms | +515 |
| head | 336 ms | 216 ms | +120 |
| experts | 1,201 ms | 1,283 ms | -82 |
| hyper-connections | 1,463 ms | 1,487 ms | -24 |
So about 2/3 of the loss is VRAM: the 8-bit dense weights take ~1,036 slots (-23%), and misses rise 21%. About 1/3 is the dense bytes. The quality is far better (0.0638 against 0.185 as shipped). On a 16 GB card the price is decode speed, which makes D2a (4 bits where the KL allows) worth doing for the PC. Raw data: G:\devel\qwen4-runs\rd-*.
TQ3 / TQ4 download tests: I can run them too if you want (disk is fine). Say so.
