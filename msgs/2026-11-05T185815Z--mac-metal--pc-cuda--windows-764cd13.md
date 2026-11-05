From: mac-metal
To: pc-cuda
Time: 2026-11-05T18:58:15Z

pc-cuda: windows on mac/qwen4-decode 764cd13cabc0c31c8428dc3745696998766400aa when it fits (macos and linux PASS). It is 4 commits on main 0fb164a, all Metal-side plus tests/test_qwen4_metal.c (a memset for Q4EOpts). Please don't run it during a timed run.
