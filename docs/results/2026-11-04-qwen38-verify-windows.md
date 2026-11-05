# Expert cache misses of MTP verify windows for Qwen3.8-Flash-Next (2026-11-04)

**Result.** A routing trace of 1,024 decode tokens of P3 went through the engine's LRU cache. The tokens were in
windows of 1 to 4. Each window is one admit, as a verify step of MTP drafts would be. The misses for each token do
not change: 117.7 to 118.0 at 8 GB, and 93.2 to 94.0 at 10 GB. So MTP drafts do not reduce the PCIe traffic of decode.
They reduce only the other costs of a step.

**Status.** Current. The trace is one text on one model. The simulation assumes that every draft is accepted. Rejected
drafts add misses.

## Setup

- Trace: `Q4E_ROUTES` of `nslm-qwen4-gen` on an HF Jobs RTX PRO 6000, job `6aebd8260e0338d1bb804e76`, kept in the
  bucket as `traces/p3-tidepools-1024.bin`. The prompt asked for a long essay on tide pools, then 1,024 greedy tokens
  with an 8 GB cache. The counts divide by the 1,025 one-row forwards.
- Model: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3).
- Simulator: `tools/qwen4_cache_sim.py --window K`. A window admits the union of the experts of K consecutive tokens in
  each layer.

```bash
python3 tools/qwen4_cache_sim.py --trace p3-tidepools-1024.bin --slots 3255 4069 --window 4
```

## Results

Decode misses for each token:

| Tokens a window | LRU, 3,255 slots (8 GB) | LRU, 4,069 slots (10 GB) | Optimum, 3,255 slots | Optimum, 4,069 slots |
|---|---|---|---|---|
| 1 | 118.0 | 93.7 | 55.3 | 41.3 |
| 2 | 117.8 | 93.6 | 55.3 | 41.2 |
| 3 | 117.9 | 93.2 | 55.2 | 41.2 |
| 4 | 117.7 | 94.0 | 55.2 | 41.1 |

## Notes

- In LRU, two tokens close together that use one expert give one miss, in one window or in two. So the misses depend on
  the sequence of routes, not on the step size.
- At 118 misses a token, P3 moves about 290 MB over PCIe for each token: about 12 ms on PCIe 4 (the RTX 4080). MTP
  cannot reduce this time. It can share the rest of a step (dense projections, attention, hyper-connections, launches)
  over the accepted tokens.
- Less PCIe traffic needs smaller experts, more cache slots, or a CPU that computes a share of the missed experts (plan
  S6). The plan's step-rate model gives the same traffic: 2.36e9 weights x 0.246 missed x 4 bits = 290 MB.
