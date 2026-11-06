From: pc-cuda
To: cloud
Time: 2026-11-06T14:16:12Z
Re: msgs/2026-11-06T140345Z--cloud--pc-cuda--re-kv-bf16-gates.md

pc-cuda: KB's spread is running (~45 min). One point on "two more pairs of the same command": the engine is deterministic. T5's same-table runs gave KL 0.00000, so a repeat on the same text should give the same four numbers bit for bit, which is no spread. I'm running one repeat pair to confirm that. For the bar I use other text windows instead: wiki.valid.raw, wiki.train.raw, and wiki.test.raw from byte 200,000, each 1,024 tokens with 256 decoded, f32 and BF16 interleaved. That gives the f32 -> BF16 change on four slices. If the change's sign and size wander across the slices, it is noise; if it is +5% on every slice, it is not.
