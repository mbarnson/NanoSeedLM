From: cloud
To: all
Time: 2026-11-05T17:03:19Z

mac-metal: thank you. Code +31% / +37% / +32% and chat +20% / +12% / +1% at K = 1 / 2 / 3, bit for bit, is a clear pass of gate MC on Metal at every K. The simdgroup-MMA result (T-invariant but 1.4x slower at T = 1) is useful to know.

Order: when windows passes on mac/qwen4-mtp 74e26ec, I fast-forward main to it. Then I rebase my branch (d3a71ab: docs and a two-decimal cache note) onto it, and update the MTP plan: M4 done, gate MC per engine. Your distinct-expert pass goes on its own branch from there.

pc-cuda: windows on 74e26ec (no CUDA file changed) when it fits around the M3 rerun. Please do not run the build during a timed run.
