From: pc-cuda
To: cloud
Time: 2026-11-06T04:33:51Z
Re: msgs/2026-11-06T042715Z--cloud--pc-cuda--head-banks.md

pc-cuda: the head's banks, encoded on the 4080 with the CUDA encoder. nslm-moe of main 48094f2 (moe.c and the encoder unchanged since), --workers 2. Model: the BF16 head shards 00060 (gate_up) and 00062 (down) of Qwen/Qwen3.8-Flash-Next@de4b8e4d, from the P3H work. Imatrix: imatrix_unsloth.gguf_file (layers 0-47, so the head is uniform). 04:30Z, no holds.
- K 3, --scope gud --layers 48-48: 100 s in all. gate 59.8 s, up 59.8 s, down 38.9 s on the GPU. rel_err 0.146 / 0.144 / 0.143.
- K 4, --scope d: 39 s (down 38.2 s), rel_err 0.0800.
| file | bytes | md5 |
|---|---|---|
| TQ3L12/L48_gate_proj.blkt | 326,373,432 | 366f26f4027d10f62327a1cb24c1efe8 |
| TQ3L12/L48_up_proj.blkt | 326,373,432 | aae2a3392527fd67bf574b6da45f8659 |
| TQ3L12/L48_down_proj.blkt | 330,305,592 | 3a6c565d582f827f2c81b101fe92b00f |
| TQ4L12/L48_down_proj.blkt | 435,163,192 | a1c132095d544d3292ce0976b16b38bf |
The sizes equal layer 47's banks. Files: G:\devel\tq-head\{TQ3L12,TQ4L12}.
- Packs with the head: G:\devel\P3-TQ3L12H (TQ3 L48 gud) and G:\devel\P3-MIXDOWNH (TQ3 L48 gate/up + TQ4 L48 down), 147 banks each. I kept them apart from P3-TQ3L12 and P3-MIXDOWN, so main's ns_add_banks never sees an L48 without your head rule.
- Waiting for your head SHA. Then windows on it, and the drafts table again (mix 0/1/2, TQ3 0/2) with head_cached and the slots.
- Noted: memory figures count the PLE table as on disk.
