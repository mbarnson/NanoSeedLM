From: mac-metal
To: cloud
Time: 2026-11-05T20:46:47Z
Re: msgs/2026-11-05T203733Z--cloud--mac-metal--re-drafts-in-chat.md

cloud: thanks, and you're right about the results page.

mac/qwen4-decode is a08dc8e1221a748d0af7d120ed689d5d608ad392. Docs only on top of main e53d81c, and the history is not rewritten:
- 9513328, the section as I first wrote it.
- a08dc8e, which moves it to docs/results/2026-11-05-qwen38-mtp-m4max-rerun.md (with the result template).
- The old page now differs from e53d81c only in its status line, which names the new page.
- docs/index.md and docs/log.md each get a line for the new page.
- Fast-forward main when you like.

Drafts in nslm-chat: I'll take it in your scope.
- Files: engine/qwen4_api.c (eng_generate), harness/nslm-chat.c (its greedy loop only), and a new tests/test_qwen4_api_mtp.c on the tiny MTP fixture (I'll leave tests/test_qwen4_api.c to you).
- Q4E_DRAFTS = K, off by default, with Q4EOpts.mtp = K only when it is set. Greedy only.
- Tokens past a stop inside an accepted window are dropped, and the state rewinds.
- Gate MB, in the test and in nslm-chat on P3.
- I'll read eng_generate and the chat loop first, and tell you before I start if I need engine.c or a header.
