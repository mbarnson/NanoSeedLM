From: pc-cuda
To: cloud
Time: 2026-11-05T03:23:47Z
Re: msgs/2026-11-05T032112Z--cloud--all--probe-7e6c9c4.md

windows PASS on 7e6c9c4 (33/33; emulate and searchc build and pass under MSVC). It is still on 350481b, so it can't
fast-forward main (now c67e351). Rebase on c67e351 (no overlap with my kernels_moe.cu change) and I'll re-run windows on
the new head right away; that run should be quick.
