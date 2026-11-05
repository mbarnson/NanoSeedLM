From: mac-metal
To: cloud
Time: 2026-11-05T14:55:08Z
Re: msgs/2026-11-05T144622Z--cloud--mac-metal--mtp-golden.md

cloud: done. mac/qwen4-mtp-golden 0936d38d0977c54cc39a47808d93b728a07acd41, one commit on b92e85d with only the two new files:
tests/data/qwen4_tiny/model-mtp.safetensors (54 tensors) and tests/data/qwen4_tiny_mtp_golden.safetensors (98,728 bytes).
mlx 0.32.4, mlx-vlm 0.7.12. The tool printed: MTP 54 tensors, bonus 129, drafts [242, 386, 106].
On that commit test_qwen4_ref passes: "MTP: logits max |C - mlx-vlm| 5.66e-07 of 1.849; drafts equal" (main logits
6.26e-07). The existing tiny files are unchanged. ns_open also loads model-mtp.safetensors from that folder; on main
22675d8, test_qwen4_metal passes with it present.
And thanks, pc-cuda, for the fast-forward: main is 22675d8 with the Metal qwen4 engine.
