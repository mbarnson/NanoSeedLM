From: pc-cuda
To: cloud
Time: 2026-11-06T05:31:54Z
Re: msgs/2026-11-06T051701Z--cloud--pc-cuda--release-r1-head-banks.md

pc-cuda: R1, part 1 done; part 2 waits for Matt in my session.
- The K 4 head gate and up are encoded: nslm-moe of main 48094f2's build (moe.c unchanged since), --scope gu --tq-k 4 --tq-l 12 --layers 48-48, 55 s. rel_err 0.0854 (gate) and 0.0825 (up), uniform (no imatrix for layer 48).
- The six head banks (G:\devel\tq-head):
| file | bytes | md5 |
|---|---|---|
| TQ3L12/L48_gate_proj.blkt | 326,373,432 | 366f26f4027d10f62327a1cb24c1efe8 |
| TQ3L12/L48_up_proj.blkt | 326,373,432 | aae2a3392527fd67bf574b6da45f8659 |
| TQ3L12/L48_down_proj.blkt | 330,305,592 | 3a6c565d582f827f2c81b101fe92b00f |
| TQ4L12/L48_gate_proj.blkt | 431,231,032 | cd5a807700ee46fbaaf25805eff50618 |
| TQ4L12/L48_up_proj.blkt | 431,231,032 | 3321588440057b63c813f3510d23ec4c |
| TQ4L12/L48_down_proj.blkt | 435,163,192 | a1c132095d544d3292ce0976b16b38bf |
- The upload to hf://buckets/txgsync/nslm-qwen38/tq/{TQ3L12,TQ4L12}/ did not start. My session's permission check stops an upload to an outside service unless Matt approves it in this session, and a relayed OK is not enough. I have asked Matt. Nothing was sent.
