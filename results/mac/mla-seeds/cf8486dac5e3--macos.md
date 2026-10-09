Status: PASS
Branch: mac/mla-seeds
Commit: cf8486dac5e3e045c7db05694e1bc450b7284cd8
Platform: macos
Agent: mac-metal
Time: 2026-10-09T03:08:38Z

make test: all tests passed. New since 1aa0fa2: seed MLA tensors decoded to BF16 once per prompt forward (prefill with
seed projections = Q8 / BF16: 8k 44.5 s), k_heads_mv two rows per simdgroup for 128-input rows, timing-mode fix for
stale sample slots, README.
