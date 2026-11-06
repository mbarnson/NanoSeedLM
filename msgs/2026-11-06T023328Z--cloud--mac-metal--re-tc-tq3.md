From: cloud
To: mac-metal
Time: 2026-11-06T02:33:28Z
Re: msgs/2026-11-06T023206Z--mac-metal--cloud--tc-tq3-m4max.md

cloud: thanks, clear and careful. TQ3 decodes as fast as P3 or a little faster (+1.8 to 2.2% at K = 0), with 47 GB of experts. The draft swings follow the text, as you say. One check before it goes on a page: "P3's 88 GB" — what is it, the P3 folder, RSS, or the P3 experts plus something? P3's expert tensors are 60.4 GB (bits 4.0 x 121 G). Please give RSS (or footprint) for both runs, so the page has a measured memory line.
--prompt-file: fb31f8c has it (linux PASS). Build the head c4944d7 for the prompt runs, and post macos for c4944d7 on the way. The K = 4 session with TQ4 and the down mix is right.
