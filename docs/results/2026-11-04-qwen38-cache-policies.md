# Expert cache policies for Qwen3.8-Flash-Next decode, from a routing trace (2026-11-04)

**Result.** On a trace of 1,024 decode tokens of P3, the engine's LRU cache of 3,255 slots (8 GB) serves 74.6% of the
expert uses. LFU and a static hot set serve fewer at every size. Belady's optimum serves 87.6%, so a policy that knows
the future could halve the misses. Only a prediction of future routes can get near it.

**Status.** Current. The trace is one text on one model. The simulator replays the engine's own LRU within 5 of its
126,208 misses.

## Setup

- Trace: `Q4E_ROUTES` of `nslm-qwen4-gen` on an HF Jobs RTX PRO 6000, job `6aebd517c78863f9c63311e3`. The prompt asked
  for a long essay on tide pools (38 tokens), then 1,024 greedy tokens with an 8 GB cache.
- Model: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3). One
  slot holds the gate, up and down slices of one expert (2.46 MB).
- Simulator: `tools/qwen4_cache_sim.py`. It counts a hit or a miss for each distinct expert of each layer and forward,
  as the engine does. The hot set holds the most used experts of the first half of the trace, and LRU manages the other
  slots.

```bash
python3 tools/qwen4_cache_sim.py --trace routes.bin --slots 2000 3255 4069
```

## Results

The share of the 497,461 expert uses that the cache serves:

| Policy | 2,000 slots (4.9 GB) | 3,255 slots (8 GB) | 4,069 slots (10 GB) |
|---|---|---|---|
| LRU (the engine) | 64.2% | 74.6% | 79.7% |
| LFU | 54.2% | 67.4% | 74.0% |
| Hot set of 25% + LRU | 39.6% | 51.1% | 57.3% |
| Hot set of 50% + LRU | 45.8% | 57.4% | 63.6% |
| Hot set of 75% + LRU | 50.2% | 62.8% | 69.0% |
| Belady (the optimum) | 80.3% | 87.6% | 90.6% |

The engine counted 371,253 hits and 126,208 misses. The simulated LRU counted 371,258 and 126,203.

## Notes

- Recent use predicts the next use better than total use does. So the plan's static hot set (S8, V3) does not help.
- At 74.6%, a token misses 123 experts: about 303 MB over PCIe. That is about 6 ms of a 22 ms token on PCIe 5 (this
  GPU), and about 12 ms on PCIe 4 (the RTX 4080).
- The next-layer prediction of `1cd7c57` raised the hit rate of the 161-token test to 82%, but its copies slowed decode.
  See [Prompt rows of 1,024 and a decode expert prediction](2026-11-04-qwen38-prompt-rows-and-prediction.md).
- Fewer bytes per expert help twice: each miss is smaller, and the same VRAM holds more slots.
