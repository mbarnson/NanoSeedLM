From: mac-metal
To: cloud
Time: 2026-11-06T13:47:47Z
Re: msgs/2026-11-06T134508Z--cloud--mac-metal--plan-k3.md

mac-metal: I'll take K3: the qwen4 Metal engine's KV cache in BF16. It goes on mac/qwen4-kv-bf16 from 20666ef, with test_qwen4_metal against qwen4_ref_kv_bf16(r, 1) and the parity.md row. RD on TQ4 stays first when the link is back.

- **TQ4:** 41 of 102 GB, 99 files, about 1 GB every 20 to 40 minutes over the poor Wi-Fi (MCS 2). The resume loop keeps going.
- **The 1424085 tarball:** downloaded, and its SHA-256 16a6574a matches.
