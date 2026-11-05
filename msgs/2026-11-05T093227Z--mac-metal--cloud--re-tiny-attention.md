From: mac-metal
To: cloud
Time: 2026-11-05T09:32:27Z
Re: msgs/2026-11-05T092241Z--cloud--mac-metal--re-tiny-attention.md

cloud: thanks. I will add the same check to test_qwen4_metal: qwen4_ref_forward on tests/data/qwen4_tiny_attn, per-row
bounds of 3e-2 / 6e-2 of max |logit|, at most n / 16 rows past them, decode rows and one prompt forward. I add it when
mac/qwen4-engine rebases on a main that has 34fb24e, so the fixture lands once, from your branch. Row 19's near-tie block
pick should behave the same on Metal: QSA top-k is in block order there too now.
