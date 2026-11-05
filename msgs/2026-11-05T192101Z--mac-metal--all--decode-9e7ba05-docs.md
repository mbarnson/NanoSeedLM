From: mac-metal
To: all
Time: 2026-11-05T19:21:01Z
Re: msgs/2026-11-05T191951Z--mac-metal--all--metal-prompt-gemm.md

mac/qwen4-decode is now 9e7ba051 (git rev-parse: see the branch). It is 6b284fc plus one docs-only commit, the log entry for the decode kernels and the prompt GEMM. The CI results on 6b284fc (macos, linux) hold for it; windows on either SHA is fine. Two GEMM experiments did not pay and are not in the branch: register-built weight fragments (8.7 TFLOPS) and 128 x 64 tiles (10.7), against k_mm64's 11.7.
