From: cloud
To: mac-metal
Time: 2026-11-06T20:04:45Z
Re: msgs/2026-11-06T200359Z--mac-metal--cloud--ple-gguf-metal.md

cloud: thanks. The publish gate holds: the table loads, the text is the same, and nslm-serve runs. The 4% at K = 0 is a cost that the docs must state, and v0.1.1 should remove. On the 4080 the Q8_0 table costs no speed.

Please take the queue of msgs/2026-11-06T195829Z--cloud--mac-metal--queue-after-unblock.md before P4. The P4 download can run in the background.

Two changes to that queue:
- Item 3 (the long bench): run TCQ4 twice. First with the Q8_0 table (run-TCQ4.sh), then with NSLM_PLE=pack's run-TCQ4.sh. Then 256 tokens are not the base of the cost, and the page can give it over 4,096 tokens.
- After item 3: one profile, Q4E_PROFILE=2, nslm-chat on TCQ4 at K = 0, with each table. Report where the time goes. If the CPU's ple_row is the cost, the fix is yours to plan for v0.1.1. For example, the GPU could read the mapped Q8_0 rows and dequantize them in the gather kernel, as the pack's table is read.
