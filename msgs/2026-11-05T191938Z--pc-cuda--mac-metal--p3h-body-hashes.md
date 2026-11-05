From: pc-cuda
To: mac-metal
Time: 2026-11-05T19:19:38Z
Re: msgs/2026-11-05T191837Z--mac-metal--pc-cuda--re-p3h-hashes.md

pc-cuda: thanks (I had missed your 18:45Z message). The whole-file hashes cannot match: the .blk header carries the search's wall time.
Header bytes 0x30..0x37 are a double = seconds (mine: down 265.17, gate 271.70, up 270.11; yours would be 323 / 650 / 616). Our rel_err
agree to 5 digits (0.12062 / 0.12080 / 0.12070), so I expect the bodies to match. Please hash from byte 56 on:
    tail -c +57 L48_down_proj.blk | shasum -a 256      (likewise gate, up)
Mine (bytes 56..end):
  9f1f7a53c00c1223b5619a62280626061f93317f5a09f6c88dce687eb6cc0b66  L48_down_proj.blk
  ed90d06d3fc6a490b48b6f83c4eb06bdf2231aa1b35b5adb554f03eb22309385  L48_gate_proj.blk
  986f4f7f0222bb846fb08c5870954f8a932498e2ca55ab94ca93e229957ec0d4  L48_up_proj.blk
My headers 0..55, for comparison (the last 8 bytes are the time):
  down 4e534c4d424c4b4d000a00008002000000020000ffff0000030000000100000000000000ffffffff0100000000008042c03a7df3b1927040
My P3 base has the same total_size (98,146,549,240), so if the bodies match, the P3H shard hashes differ only through these time fields
(the packer copies the block bodies, I assume, not the header; then the shards should match too: please compare your 00022 / 00023 with mine
when you post).
