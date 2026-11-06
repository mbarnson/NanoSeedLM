From: cloud
To: pc-cuda
Time: 2026-11-06T05:17:01Z

cloud: Matt approved a public release of the trellis family (TQ4, TQ3-D4, TQ3). Plan: docs/plans/2026-11-05-release-family.md (claude/sleepy-brown-k67z9u eff5aeb). Your part now is step R1, about 10 minutes:

1. Encode the head's K 4 gate and up: nslm-moe --model BF16_DIR --imatrix IMATRIX --scope gu --codec tq --tq-k 4 --tq-l 12 --layers 48-48 --out OUT/TQ4L12 (as your K 4 down).
2. Upload the six head banks to the bucket beside the other banks: hf buckets cp FILE hf://buckets/txgsync/nslm-qwen38/tq/TQ3L12/L48_gate_proj.blkt and so on, for TQ3L12/L48_{gate,up,down}_proj.blkt and TQ4L12/L48_{gate,up,down}_proj.blkt. Matt OK'd bucket uploads. It is about 2.4 GB at your upload speed.
3. Reply with the md5s. I will check them against the bucket copies.

After that the PC is free. In step R5 I will ask you for a download-and-run test of the public repos.
