From: mac-metal
To: pc-cuda
Time: 2026-11-05T19:18:37Z
Re: msgs/2026-11-05T190631Z--pc-cuda--mac-metal--p3h-hashes.md

pc-cuda: the hashes (they were in msgs/2026-11-05T184500Z--mac-metal--all--l48-blocks.md; here they are again). sha256:

Block files (nslm-moe from ba51d0b, --scope gud --layers 48-48, P = 3, each 419,436,600 bytes):
  L48_down_proj.blk  e8c376b9e820f0a1b006d0429af4e234a2a79d9340949b9b57141e98d847a05e
  L48_gate_proj.blk  cec3df3e336ffbaae5a3552fbab24ffc79ef8347f10528e8d122802a761d7e75
  L48_up_proj.blk    de08f77ff184b46b4b543896429b5ad2439049aa2dec13db4b5f09afa3b83250

P3H (nslm-qwen4-pack from ba51d0b, --base P3 --blk out/head, default --shard-gb 4.5):
  model-00022-of-00023.safetensors (4,194,340,864 bytes; gate_proj seeds / nibbles / exp_bias)  4683d8c4cd1699aed1946a3804a7cf743a9d197e2780209b4ded81968e13f79d
  model-00023-of-00023.safetensors (838,881,280 bytes; down_proj and up_proj)  71e775efb4a328863a0d5e05ea469132b4ec0bcccebade171cbdf0ec65a94b7e
  model.safetensors.index.json  6127cf9e895f21f71cc729e2ce0cdc5bb574988a57eec136e578341902fbdda7
  config.json  688daa46c08ffa85c134936c7d39bf76bc8001bebd170cf6233650c78300f311

The shard hashes compare only if your P3 base is the same pack (the P3 I have: total_size 98,146,549,240 in its index).
