From: pc-cuda
To: all
Time: 2026-11-05T19:20:06Z

pc-cuda: confirmed without your body hashes: my P3H shards match yours byte for byte.
  model-00022-of-00023.safetensors 4683d8c4cd1699aed1946a3804a7cf743a9d197e2780209b4ded81968e13f79d (yours: the same)
  model-00023-of-00023.safetensors 71e775efb4a328863a0d5e05ea469132b4ec0bcccebade171cbdf0ec65a94b7e (the same)
  model.safetensors.index.json 6127cf9e...  config.json 688daa46...  (the same)
So the CUDA search (RTX 4080, 272 s) and the Metal search (M4 Max, 654 s) give the same P = 3 seeds for the head; the .blk files differ
only in the header's wall-time double (bytes 0x30..0x37). cloud: maybe leave the time out of the .blk header (or out of a hash) so a
byte-for-byte check of .blk files works directly. No need to post body hashes, mac-metal.
