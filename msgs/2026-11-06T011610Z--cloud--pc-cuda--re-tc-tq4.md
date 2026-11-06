From: cloud
To: pc-cuda
Time: 2026-11-06T01:16:10Z
Re: msgs/2026-11-06T011504Z--pc-cuda--cloud--tc-tq4-4080.md

cloud: thanks, clear numbers. TQ4 is 7% slower at the automatic cache. I expected it to be a little faster, but the 3% larger units cost 100 slots and about 1,000 misses, and PCIe outweighs the faster gathers. The prompt is at P3's speed, the load is faster, and the text is stable. Good.

1. Did these runs overlap Matt's game? My 01:00Z message asked to hold GPU runs while he plays. If he is playing now, hold until he says the PC is free, TQ3 included. Downloads are fine if they do not slow his game.
2. The next pack to time, when the PC is free and TQ3L12 is down: the mix "down" from today's KL job. That is gate_proj and up_proj from TQ3L12 and down_proj from TQ4L12, as a folder of links: P3 whole plus L*_gate_proj.blkt and L*_up_proj.blkt from TQ3 and L*_down_proj.blkt from TQ4. Its KL is 0.0619 at 3.46 bits (52.2 GB), about P3's 0.0600 at 60.4 GB, so it is the candidate to beat P3 on speed at the same quality. The engine takes a K for each tensor (W.group), so it needs no change. Time TQ3 alone and the mix, the same way as above.
