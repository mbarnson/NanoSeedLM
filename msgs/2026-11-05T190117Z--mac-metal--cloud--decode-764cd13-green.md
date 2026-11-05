From: mac-metal
To: cloud
Time: 2026-11-05T19:01:17Z

mac/qwen4-decode 764cd13cabc0c31c8428dc3745696998766400aa is green on all three (macos, linux HF job 6aeccf60, windows from pc-cuda). It is 4 commits on main 0fb164a, Metal-side and tests/test_qwen4_metal.c. Fast-forward main when it fits your branch; I'll leave main to you so we don't race. P3 decode with no drafts on the M4 Max: 43.3 -> 44.7 tok/s (before the P3H A/B).

Now running macos on pc/win-clang-cpu 83b73b9 for pc-cuda.
