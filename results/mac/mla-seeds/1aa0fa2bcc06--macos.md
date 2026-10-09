Status: PASS
Branch: mac/mla-seeds
Commit: 1aa0fa2bcc06e1d4b5a01ab3f15e907acd5d8eb1
Platform: macos
Agent: mac-metal
Time: 2026-10-09T02:58:25Z

make test: all tests passed (rebased onto main ca86b47; per-head SEED4P4 kernel tests, MLA capture test).
KLD (naive J768 p4mx-q4v, 31 windows, bf16 KV): --mla p4 0.2296; sensitivity (one group kept Q8): rope 0.2297,
kv_a_x+kv_a_v 0.2259, q_lat 0.2271, v_up 0.2250; all Q8 0.2206.
