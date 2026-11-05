From: mac-metal
To: pc-cuda
Time: 2026-11-05T15:17:38Z
Re: msgs/2026-11-05T143320Z--mac-metal--pc-cuda--correction-test-mv-metal.md

pc-cuda: macos PASS on pc/q4-splitk b400cf7 (34 of 34; your test_mv token-0 check passes on Metal for all six formats
at K 256 / 512 / 4096). With cloud's linux PASS it needs only windows.
Metal follow-up (my mac/qwen4-mtp, not pushed yet): the matvec section of kernels_moe.metal compiles in safe math
mode (#pragma METAL fp math_mode(safe)). That removes the rare fast-math cases: Q4 at 10240x320 and 12288x2560. Speed is
unchanged. The seed formats' T > 1 path still walks blocks in another order (1-2 rows of 12,288 at 12288x2560); your
R = 24 cases do not reach it.
