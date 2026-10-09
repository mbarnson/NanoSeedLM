Status: PASS
Branch: mac/mla-pack
Commit: 0a3d437cae6f6b9c2c0124aa61b491feb6830878
Platform: macos
Agent: mac-metal
Time: 2026-10-09T01:20:24Z

make test: all tests passed (test_model_st: affine tensors without .weight names; per-head Q8/Q4 kernels).
KLD (naive J768 p4mx-q4v, 31 held-out windows): bf16 projections 0.2199, --mla q8 0.2206, q4 0.2471, q8 + fp4 KV 0.2246.
