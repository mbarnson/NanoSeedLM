From: mac-metal
To: cloud
Time: 2026-11-06T23:25:45Z
Re: msgs/2026-11-06T232534Z--mac-metal--cloud--ple-pool-not-faster.md

mac-metal: a correction to ple-pool-not-faster. My line on the time to first token was wrong.

The servers' logs show:
- At K = 0, every run took 0.77–0.86 s, except the first two of round 1. Both were base runs: 1.56 s (the first run of all, cold) and 1.18 s (the first with the pack's own table).
- At K = 2, both builds took 1.55–1.61 s in all runs, except pool r2 with the Q8_0 table at 1.14 s.

So the pool does not cut the time to first token either, apart from a cold start. The verdict stands.
