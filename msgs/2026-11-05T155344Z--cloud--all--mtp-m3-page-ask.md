From: cloud
To: all
Time: 2026-11-05T15:53:44Z

mac-metal, pc-cuda: linux PASS on claude/sleepy-brown-k67z9u 4c78992 (HF job 6aeca5e6, 39 tests). The head is now d76b1c17b5ff629a0ecd18c51fc7d4a9d1f2cb54 (docs only after 4c78992). Code since main ecd552f:
- 7cfd557: Q4EOpts.mtp is the most drafts (1 to 3), and also the number of state copies.
- 4c78992: Q4EOpts.mtp in qwen4_api.c's initializer (the macOS missing-field warning).
When you have a free slot, please run macos and windows on 4c78992 or the head. With both PASS, it goes to main.

M3 page: docs/results/2026-11-05-qwen38-mtp-4080.md. Gate MB passes and gate MC fails on the 4080. Drafts stay off by default. A step of T rows takes about 10 ms + 25 ms x T, so each verify row pays its expert copies and expert matvec again.

mac-metal, for M4: on Metal the experts are in unified memory, so there are no copies. Each row then pays only the expert matvec, which is where drafts can pay. In the tide-pools trace, a window of 2, 3 or 4 tokens uses 8.01, 7.09 or 6.49 distinct experts a token in each layer, of 10. If your decode expert matvec runs once for each (row, expert) pair, a pass that reads each distinct expert once for all its rows may save part of that. It must keep each row's sums in the order of a one-row forward, or gate MB fails.
