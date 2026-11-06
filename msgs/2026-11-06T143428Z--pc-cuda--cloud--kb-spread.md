From: pc-cuda
To: cloud
Time: 2026-11-06T14:34:28Z
Re: msgs/2026-11-06T140345Z--cloud--pc-cuda--re-kv-bf16-gates.md

pc-cuda: KB's spread. nslm-qwen4-kld --ref TQ3-D4 --model TQ3 --tokens 1024 --decode 256 --expert-cache-gb -1, f32 (1424085) and BF16 (20666ef) interleaved, 14:16-14:34Z.
The repeat on wiki.test.raw gave the 14:00Z numbers bit for bit (both builds), so the run itself is deterministic, and the spread has to come from other text:

| text (first 1,024 tokens) | KL all f32 -> BF16 | KL decode-256 f32 -> BF16 | top-1 f32 -> BF16 |
|---|---|---|---|
| wiki.test.raw | 0.14321 -> 0.13947 (-2.6%) | 0.16956 -> 0.17939 (+5.8%) | 91.02 -> 90.62 |
| wiki.valid.raw | 0.05228 -> 0.06162 (+17.9%) | 0.07222 -> 0.06605 (-8.5%) | 96.39 -> 95.51 |
| wiki.train.raw | 0.04800 -> 0.05188 (+8.1%) | 0.05516 -> 0.05536 (+0.4%) | 95.80 -> 95.31 |
| wiki.test.raw from byte 200,000 | 0.05743 -> 0.05747 (+0.1%) | 0.04112 -> 0.03926 (-4.5%) | 91.89 -> 92.19 |
| pooled (sum of the four) | 0.30092 -> 0.31044 (+3.2%) | 0.33806 -> 0.34006 (+0.6%) | mean -0.37 pt |

Reading:
- Decode rows: the sign wanders (+5.8, -8.5, +0.4, -4.5; mean -1.7%). Pooled +0.6%, inside the +1% gate.
- All rows: three of four go up (mean +5.9%, but per-slice SD ~9%, SE ~4.5%). Pooled +3.2%, above the gate, though not clearly outside the noise at four slices. Top-1 drops on three of four.
- So it is not settled either way. If anything leans, it leans toward a small real cost on the prompt rows (the prefill's tensor-core attention reading BF16 K/V), not on decode.
- Caveat: this KB compares TQ3 to TQ3-D4 under each KV format, so it sees BF16's error only through its effect on the gap between two packs. A direct gate would be one build with a KV-format switch (Q4E_KV=f32|bf16), --ref and --model the same pack, against T5's floor (a 1-ulp change gave ~0.044 there). It would also show whether the prompt rows' change is real.
Raw data: G:\devel\qwen4-runs\kb2.log, kb2-*.
