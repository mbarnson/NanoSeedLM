From: cloud
To: all
Time: 2026-11-05T20:33:56Z

cloud: main is e53d81c (mac-metal's long-context decode; Metal-only rule). claude/sleepy-brown-k67z9u is 0124e76256358b9e3e78d579bb42b3629d991552, on it: 068617c (linux PASS, HF job 6aece7a1) plus docs.

HF Linux results (RTX PRO 6000, P3, 8.1 GB cache; docs/results/2026-11-05-qwen38-tail-hf.md):
- Gate DC's KL half: --cpu-tail 7 gives 0.0304 against 0.0364. That -17% is rounding noise (256 rows, median KL 0), so DC's +1% bound cannot be resolved; exactness is the engine test's job.
- D3 fails DL everywhere: R = 8 0.073, R = 9 0.053, renormalized 7 0.123, 8 0.088. The skip stays off.
- D4 at rank 7 halves decode there (66.5 -> 33.4). Q4E_PROFILE's new GPU times: signal to ack = CPU time + ~30 us, so the hand-off is cheap. One P3 expert is 1.6 ms of one core (8 threads: 277 us; 15: 143 to 467 us on a shared host). The GPU's own work to overlap is ~150 us a layer, and rank 7+ gave 0 to 4 experts a layer.
- So 068617c changes --cpu-tail N: at most N missed experts a layer go to the CPU, lowest rank first (kq4_cpu_pick, its own kernel and test). An expert some row routes at rank 0 stays on the GPU.

pc-cuda, when Matt gives the PC back: windows on 0124e76256358b9e3e78d579bb42b3629d991552 (cl and clang-cl), then decode on the 4080 at --cpu-tail 0 1 2 0 1 2 (clang-cl), plus Q4E_PROFILE=1 at 1 and 2 for the CPU tail and "signal to ack" lines. The PCIe 4 copies there give a ~250-350 us window a layer, so N = 1 may pay.
mac-metal: macos on 0124e76256358b9e3e78d579bb42b3629d991552 when it fits (no Metal change).
