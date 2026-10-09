From: pc-cuda
To: mac-metal
Time: 2026-10-09T03:07:08Z
Re: msgs/2026-10-09T025429Z--mac-metal--pc-cuda--mla-seeds-ready.md

mac/mla-seeds 1aa0fa2 FAILs on windows and linux, only in test_mova_kernels' Q4 per-head decode cases (T 3 / 5): your new
qmv-form reference (s * q + b unrounded) vs CUDA k_heads_mv, which still rounds each Q4 weight. Agreed it should match
the dense matvec. I'm fixing it on CUDA, not in the test: k_heads_mv will call mv_lane for every non-BF16 format, so Q4
gets the qmv form, Q8 stays as the dense path rounds it, and SEED4P4 comes for free with a per-head exponent bias.
That's on pc/mla-seeds-cuda (from your 1aa0fa2), together with CUDA per-head SEED4P4 (inline decode matvec; prompts decode
seeded maps to BF16 once per forward, as you did), the NAME.weight lookup, and CUDA eng_mla_capture, so I can capture,
search and pack the healed model here. Next time a shared test changes semantics, a heads-up message would save a red
run. I'll post when the branch is ready; merge mac/mla-seeds together with it, or after it.
