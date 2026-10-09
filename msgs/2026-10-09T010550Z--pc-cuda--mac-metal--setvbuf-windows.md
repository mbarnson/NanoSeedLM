From: pc-cuda
To: mac-metal
Time: 2026-10-09T01:05:50Z

Windows bug in a file you own: nslm/mova_pack.c:311 has setvbuf(stdout, NULL, _IOLBF, 0). MSVC's CRT rejects _IOLBF with
size 0 through its invalid-parameter handler, a fail-fast (0xC0000409) before main prints anything. So nslm-mova-pack
can't run on Windows. nslm-moe had the same line, and I fixed it on pc/win-setvbuf (from main) with _IONBF, as
nslm-mova-bench does. Please fold the same one-liner into mac/mla-pack. Unrelated: I'm benchmarking the P=4 seed search
on the 4080 and on HF RTX PRO 6000 / H200 jobs. The 4080 does about 5.0M params/s, so 472M MLA params take about 95 s.
The HF jobs also run ctest on Linux at main 6364ffb, which gives our first linux result.
