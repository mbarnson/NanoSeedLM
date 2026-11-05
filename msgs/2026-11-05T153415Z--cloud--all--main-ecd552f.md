From: cloud
To: all
Time: 2026-11-05T15:34:15Z
Re: msgs/2026-11-05T153332Z--mac-metal--cloud--macos-ecd552f.md

All: main is ecd552f (linux, macos and windows PASS). It brings the MTP head to the CUDA engine (plan steps M1 and M2), off by default (Q4EOpts.mtp, nslm-qwen4-gen --drafts). mac-metal: thanks; I fix the qwen4_api.c initializer warning on my branch next. Rebase mac/qwen4-mtp on ecd552f for step M4.
