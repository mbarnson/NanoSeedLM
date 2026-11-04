# pc-cuda claims (updated 2026-11-04T22:22:20Z)

- **Now (2026-11-04, Matt's ask):** testing `claude/sleepy-brown-k67z9u` (PR #15, Qwen3.8-Flash-Next CUDA engine) on the
  4080: ctest, then nslm-qwen4-gen on the P3 pack (default pinned host experts, --expert-cache-gb 4/8/10), then P4.
  Checkout G:\devel\NanoSeedLM, packs in G:\devel. No code pushes; report to out/qwen4-4080-report.md (local).
- Queue (CUDA files): fp4 decode gap vs BF16; KV VRAM rows by active slot; full-model GPTQ capture when Matt is around.
