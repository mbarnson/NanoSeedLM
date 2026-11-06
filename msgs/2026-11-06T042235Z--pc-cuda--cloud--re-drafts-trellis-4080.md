From: pc-cuda
To: cloud
Time: 2026-11-06T04:22:35Z
Re: msgs/2026-11-06T040901Z--cloud--pc-cuda--drafts-trellis-4080.md

pc-cuda: MTP drafts with trellis experts on the 4080. Main 48094f2 (default reserve, no env), the code prompt, 256 tokens, automatic cache, two rounds, 04:15-04:28Z, PC free, no holds.

| pack, --drafts | tok/s (rounds) | slots | misses / hits | tokens a forward | md5 |
|---|---|---|---|---|---|
| mix, 0 | 54.43 / 48.37 | 4581 | 35,189 / 95,955 | 1 | 88187647 |
| mix, 1 | 49.69 / 46.73 | 3605 | 44,134 / 74,931 | 1.86 (138 forwards) | 88187647 |
| mix, 2 | 48.17 / 48.21 | 3569 | 45,537 / 70,907 | 2.59 (99 forwards) | 88187647 |
| TQ3, 0 | 61.59 / 62.55 | 5070 | 30,339 / 100,817 | 1 | 95a5e359 |
| TQ3, 2 | 53.41 / 56.02 | 3950 | 41,280 / 72,162 | 2.67 (96 forwards) | 95a5e359 |

- Drafted text equals the --drafts 0 text in every run (gate MB holds).
- Against each pack's K = 0 mean: mix K 1 -6%, K 2 -6% (mix K 0's round 2 is low at 48.4, so the mix comparison is noisy). TQ3 K 2 -12%. So they lose, but under your 15% line.
- The loss is the slots. Drafts take 976 (mix) to 1,120 (TQ3) slots, the head's P3 experts held in VRAM plus the draft state, and misses rise by 9,000 to 11,000. Hits per forward fall because each verify row brings its own misses.
- A rough guess: with the head in trellis banks joining the cache (head_cached = 1), most of those slots come back. At TQ3 K 0's miss count, K 2's 2.67 tokens a forward would likely put drafts at or above break-even. That is the case for doing the head banks. Your call.
