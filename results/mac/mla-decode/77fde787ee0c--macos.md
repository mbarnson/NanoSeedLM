Status: PASS
Branch: mac/mla-decode
Commit: 77fde787ee0c11dee0a6ea3d18f2b5dead6800b7
Platform: macos
Agent: mac-metal
Time: 2026-10-09T01:32:18Z

make test: all tests passed. Metal-only (kernels_moe.metal): register-blocked k_mla_attn scores; 8k decode attention
248 -> 214 us per layer; P=4 decode at 8k 29.8 -> 31.1 tok/s.
