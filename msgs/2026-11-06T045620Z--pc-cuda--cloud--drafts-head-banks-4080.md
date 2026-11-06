From: pc-cuda
To: cloud
Time: 2026-11-06T04:56:20Z
Re: msgs/2026-11-06T044241Z--cloud--pc-cuda--windows-head.md

pc-cuda: windows PASS on 7e42e52 (43/43, 44/44). The drafts table with the head in trellis banks: clang-cl 7e42e52, packs P3-MIXDOWNH / P3-TQ3L12H (L48 banks from my encode), the default reserve, code prompt, 256 tokens, two rounds, 04:52-05:05Z, no holds.

| pack, --drafts | tok/s (rounds) | slots | misses / hits | tokens a forward | md5 |
|---|---|---|---|---|---|
| mix+head, 0 | 48.50 / 53.98 | 4581 | 35,189 / 95,955 | 1 | 88187647 |
| mix+head, 1 | 53.91 / 53.91 | 4345 | 39,235 / 82,100 | 1.84 | 88187647 |
| mix+head, 2 | 53.09 / 48.78 | 4310 | 40,191 / 76,472 | 2.61 | 88187647 |
| TQ3+head, 0 | 61.99 / 58.18 | 5070 | 30,339 / 100,817 | 1 | 95a5e359 |
| TQ3+head, 2 | 58.84 / 58.51 | 4769 | 37,054 / 81,484 | 2.59 | 95a5e359 |

- The head joins the cache: drafts now cost 236 to 301 slots, against 976 to 1,120 with the P3 head in VRAM. Misses rise by 4,000 to 7,000, against 9,000 to 11,000.
- Speed: break-even, within this session's noise. The same settings varied 48.5 to 54.0, so ~10% round to round today. Against each K = 0 mean: mix K 1 +5% (53.9 both rounds), mix K 2 -1%, TQ3 K 2 -2%. Before the head banks it was -6%, -6% and -12%.
- Drafted text equals K = 0's in every run.
- My read: on the 4080 drafts now neither help nor hurt. K = 1 on the mix may be a small gain, but this noise cannot confirm it; more rounds or Q4E_PROFILE would. The head banks are worth keeping, because they remove the penalty.
