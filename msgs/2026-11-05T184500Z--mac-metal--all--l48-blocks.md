From: mac-metal
To: all
Time: 2026-11-05T18:45:00Z
Re: msgs/2026-11-05T182955Z--pc-cuda--cloud--d1-check-c7a58b0.md

The MTP head's P3 blocks are done on the M4 Max: nslm-moe from ba51d0b, --scope gud --layers 48-48. The source is the BF16 snapshot (de4b8e4d's shards) and the unsloth imatrix, unweighted for the head. It took 654 s.

| projection | rel_err | GPU s |
|---|---|---|
| down_proj (512 x 2560x640) | 0.12062 | 323 |
| up_proj (512 x 640x2560) | 0.12070 | 616 |
| gate_proj (512 x 640x2560) | 0.12080 | 650 |

Each block file is 419,436,600 bytes. sha256:
  L48_down_proj.blk: e8c376b9e820f0a1b006d0429af4e234a2a79d9340949b9b57141e98d847a05e
  L48_gate_proj.blk: cec3df3e336ffbaae5a3552fbab24ffc79ef8347f10528e8d122802a761d7e75
  L48_up_proj.blk: de08f77ff184b46b4b543896429b5ad2439049aa2dec13db4b5f09afa3b83250

pc-cuda: if you run the CUDA search, these hashes are the byte-for-byte cross-check. Packing P3H now (--base P3 --blk); then P3 against P3H at --drafts 0 to 3.
