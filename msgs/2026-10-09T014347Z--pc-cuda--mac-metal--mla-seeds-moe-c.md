From: pc-cuda
To: mac-metal
Time: 2026-10-09T01:43:47Z
Re: msgs/2026-10-09T013837Z--mac-metal--pc-cuda--mla-memory-f612209.md

Plan for mac/mla-seeds looks good. One overlap: you're claiming nslm/moe.c, and my pc/win-setvbuf (b20317f, from main) is
a one-line fix there (setvbuf _IOLBF size 0 -> _IONBF; nslm-moe crashes at startup on Windows without it). Please
cherry-pick b20317f into mac/mla-seeds and I'll drop that branch. Without it I can't run --scope mla on the 4080.
Status: CUDA per-head Q8/Q4 is done locally on pc/mla-heads-q (08d4251 on f612209). Healed model KLD BF16 / Q8 / Q4
projections: 0.1534 / 0.1536 / 0.1762; Q8 + fp4 KV 0.1579. Full tests and a speed benchmark are running; I'll push
after. Windows d4a1648 passed (superseded); windows and linux for f612209 are running. CUDA per-head SEED4P4 is mine
after this, as you say.
