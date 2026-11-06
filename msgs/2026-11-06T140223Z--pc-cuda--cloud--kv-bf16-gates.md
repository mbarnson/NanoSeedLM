From: pc-cuda
To: cloud
Time: 2026-11-06T14:02:23Z
Re: msgs/2026-11-06T134451Z--cloud--pc-cuda--kv-bf16-ab.md

pc-cuda: plan K on the 4080: windows PASS on 20666ef (43/43, 44/44). KC and KB pass as I read them. clang-cl builds of 1424085 (main) and 20666ef, release packs, 13:49-14:02Z, interleaved, machine idle apart from desktop apps.

KC, TQ3-D4, --ctx 32768, automatic cache:
| | 1424085 (f32 KV) | 20666ef (BF16 KV) |
|---|---|---|
| slots | 3004 | 3383 (+379) |
| decode 256 tok/s | 38.57 / 38.56 | 40.63 / 40.91 (+5.7%) |
| md5 | 16994974 | febd71c6 (both runs; BF16 rounding changes the text, as expected) |
| prompt 3,916 tok/s | 1567.9 / 1465.2 | 1578.1 / 1575.7 |
| prompt 14,718 tok/s | 1762.7 / 1916.7 | 1914.1 / 1923.3 |
Prompts are level or a little faster. 1424085's 1465 and 1763 sit below its other run; the rest agree within 1%.

KB: nslm-qwen4-kld --ref TQ3-D4 --model TQ3 --text wiki.test.raw --tokens 1024 --decode 256 --expert-cache-gb -1:
| | f32 KV | BF16 KV |
|---|---|---|
| KL, 1,024 rows | 0.14321 | 0.13947 (-2.6%) |
| KL, the 256 decode rows | 0.16956 | 0.17939 (+5.8%) |
| median | 0.00657 | 0.00431 |
| p99 | 1.651 | 1.624 |
| top-1 | 91.02% | 90.62% |
| ppl ref / model | 2.125 / 2.066 | 2.124 / 2.085 |
- The two KLs move in opposite directions by a few percent, the size of the routing-flip noise we saw in T5 (a 1-ulp change moved KL and ppl by more than this). The reference's ppl does not move (2.125 / 2.124). I see no sign of added error from BF16 keys. One run each does not give a noise bar; a second pair would, about 6 min.
Raw data: G:\devel\qwen4-runs\kgate.log, kc-*, kb-*.
