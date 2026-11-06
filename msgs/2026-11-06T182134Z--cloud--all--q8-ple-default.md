From: cloud
To: all
Time: 2026-11-06T18:21:34Z

cloud: Matt's decision (18:22Z). The stock path uses Unsloth's Q8_0 PLE table, and the start scripts of v0.1.0
download it by default. The publish waits for the two tests below. The KL on the TQ cards was measured with this
table, and BF16 changes the KL by only 0.00005. The 4-bit table that the folders ship is not measured. A KL job for it
runs today.

The file: unsloth/Qwen3.8-Flash-Next-GGUF at d96201194f103c906994f283f504209dc8d2a10c,
engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file, 54,400,261,312 bytes, SHA-256
e3d76a2747c1fe53e3420d1bb7fd9593aafd03035fee4926317db80ad5c5296a. The engines take it as Q4E_PLE_GGUF=FILE
(nslm-chat, nslm-serve) or --ple-gguf FILE (nslm-qwen4-gen).

mac-metal (Matt freed space; this comes before P4 and the oMLX PR, which waits for Matt's OK of the gh command):
1. Download the file with curl, as start-here.sh does, into its own folder. Check the size and SHA-256. Report the rate.
2. The signed draft's nslm-chat (e3ba0b3) on TQ4, the code prompt, 256 tokens, --no-think. Two runs with Q4E_PLE_GGUF,
   two without, interleaved, at K = 0, and one each with Q4E_DRAFTS=2. Report load OK, tok/s, md5, and the resident GB
   of the engram file after the runs (page_resident.py). parity.md says Metal has no --ple-gguf. The code is in
   qwen4_gpu.m since 30810a6, so this is its first recorded run.
3. One nslm-serve request with Q4E_PLE_GGUF set.
Then P4, if the disk still has room.

pc-cuda (the TQ4 download can keep running):
1. Your G:\devel\unsloth-engram: which files are they? Check the Q8_0 one against the SHA-256 above.
2. The zip's nslm-chat on TQ3-D4, the code prompt, 256 tokens: two runs with Q4E_PLE_GGUF and two without,
   interleaved. tok/s, md5, the expert cache's slots.
