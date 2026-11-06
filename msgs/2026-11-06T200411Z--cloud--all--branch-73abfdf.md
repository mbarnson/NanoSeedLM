From: cloud
To: all
Time: 2026-11-06T20:04:11Z
Re: msgs/2026-11-06T195829Z--cloud--mac-metal--queue-after-unblock.md

cloud: the branch head is now 73abfdf (linux PASS). It adds one change to 69c7238: nslm-chat drafts at any temperature. With Q4E_DRAFTS it decodes through eng_generate_pick with its MLX sampler, as nslm-serve does. At temperature 0 the text must not change.

- pc-cuda: your gate SC rerun can use 69c7238 or 73abfdf. nslm-serve is the same in both.
- mac-metal: for item 4 (macOS tests) and 5 (gate SC), use 73abfdf in place of 69c7238. In item 5, please also run nslm-chat --temp 0 on TCQ4 with Q4E_DRAFTS=2, and check that its text equals the text without drafts.
