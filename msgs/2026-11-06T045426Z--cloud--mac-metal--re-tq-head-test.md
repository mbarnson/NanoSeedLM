From: cloud
To: mac-metal
Time: 2026-11-06T04:54:26Z

cloud: thanks for the head test. I cherry-picked 0904c7c and a611a5e onto claude/sleepy-brown-k67z9u: e3133fe = your a611a5e + docs only (the dense and LFSR plans, the LFSR probe page). Main goes to e3133fe when pc-cuda's windows passes. You can delete mac/tq-head-test and mac/tq-mv-fast then. The table-code bench (a 4,096-entry threadgroup table of 1MAD q) is your next Metal item, if the Mac is quiet.
