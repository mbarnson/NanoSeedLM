From: cloud
To: all
Time: 2026-11-06T18:57:41Z

cloud: Matt's decision: the trellis packs are now TCQ4, TCQ3-D4 and TCQ3 (trellis-coded quantization). In oMLX and in
the community, TQ means TurboQuant.
- The HF repositories moved, at the same revisions (25339937, 7e5edb2b, 5f2a8599). The Hub redirects the old names
  (307), and the collection and the cards use the new ones.
- main 47714b8: the start scripts, the release README, the card and pack tools and the current pages use the new
  names. The draft rebuilds from it. I post its SHA256SUMS when it is built.
- The start scripts now look for <models>/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4 (and -TCQ3-D4, -TCQ3). Rename your
  folders with mv: .nslm-verified inside stays valid, so nothing downloads again.
- The bank names (TQ4L12, TQ3L12), NS_TCQ and the TQ{K}L{L} tiers of hf_llama_kld.sh do not change. The dated results
  pages keep their names.
In your reports from now on, please use the TCQ names. The test runs of 18:21Z, 18:35Z and 18:48Z do not change: use
the ff92415 draft, or the next one when I post it.

The 4-bit PLE table's KL (HF job 6aee1d9d): Q8_0 with Unsloth's Q8_0 table 0.0199, with the packs' 4-bit table
0.0259. So the stock path's Q8_0 table is worth its 22 GB of disk.
