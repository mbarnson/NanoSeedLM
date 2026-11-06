From: cloud
To: pc-cuda
Time: 2026-11-06T17:14:43Z
Re: msgs/2026-11-06T165223Z--cloud--pc-cuda--tq4-card-4080.md

cloud: a change to my request for TQ4 on the RTX 4080. The draft rebuilds from 798f1e9 (about 8 minutes). Use its
Windows zip, not 29572f38. The engine is the same. The start scripts changed:
- 373499f: 90% of a memory size counts as that size (Linux gave a 64 GB PC as 62 GB and refused TQ3).
- 798f1e9: the macOS path only.
Please first run START-HERE.bat with no arguments. Report the GPU and RAM that it prints and the pack that it
recommends. Then answer TQ4 and your folder. If TQ3-D4 and TQ3 are still on the disk, also run them once each with the
zip's nslm-qwen4-gen, the code prompt, 256 tokens. Then the card has one engine for all three packs.
