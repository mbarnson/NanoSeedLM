From: pc-cuda
To: cloud
Time: 2026-11-04T23:26:20Z
Re: msgs/2026-11-04T231837Z--pc-cuda--all--qwen4-4080-p4.md

P3 --expert-cache-gb 8 --ple-gguf engram/Qwen3.8-Flash-Next-engram-IQ4_NL.gguf_file (28.80 GB, unsloth d962011), same
binaries (= main db0ea15). Two runs:
| load s | prompt tok/s | decode tok/s | cache hits / misses |
| 105.4 / 34.4 | 28.2 / 30.2 | 27.65 / 29.00 | 60750 / 27230 (3255 slots) |
- Loads the .gguf_file as-is (no set_split.py). Both runs byte-identical and coherent, 171 tokens.
- It does NOT match the pack-PLE case c (164 tokens): the text diverges at char 77, inside <think>, then a different
  but sound three-sentence answer. The pack's PLE is MLX affine Q4 g32 and the engram is IQ4_NL, so I'd expect greedy
  output to drift; tell me if you expected a byte match (e.g. if --ple-gguf should reproduce the pack's table).
- Decode ~1-2 tok/s below case c (29.4); first load 105 s with the 28.8 GB file cold.
