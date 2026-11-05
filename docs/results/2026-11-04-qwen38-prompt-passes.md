# Layer-major prompt passes for Qwen3.8-Flash-Next on CUDA (2026-11-04)

**Result.** The test used an RTX PRO 6000 with an 8 GB expert cache. Layer-major passes take a 3,703-token prompt of
P4 from 536 to 833 tokens per second, and a 15,922-token prompt from 549 to 977. The cache misses of the long prompt go
from 270k to 50k. The generated text does not change.

**Status.** Current. The RTX PRO 6000 has PCIe 5. The RTX 4080 (PCIe 4) is not measured.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell, 96 GB), job `6aebe75dc78863f9c63318ec`. The 1-minute load average
  was 3.0 to 8.6 during the runs.
- Model: [Qwen3.8-Flash-Next-NanoSeedLM-P4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4).
- Builds: A is `25ae5f9` (forwards of up to 1,024 rows). B is `4e6d2d7` (layer-major passes of up to 8,192 rows).
- Prompts: the first 16,000 and 64,000 characters of the wikitext-2 test set, raw. Two runs of A and B in turn.

```bash
nslm-qwen4-gen --model PACK --prompt "$(cat prompt.txt)" --raw --max-tokens 32 --ctx 16384 --expert-cache-gb 8
```

## Results

| Prompt | Build | Prompt, tokens/s | Cache hits / misses | Decode, tokens/s |
|---|---|---|---|---|
| 3,703 tokens | A | 530.8 / 541.8 | 11,050 / 69,007 | 30.95 / 31.19 |
| 3,703 tokens | B | 818.0 / 848.0 | 54,670 / 25,387 | 31.08 / 31.17 |
| 15,922 tokens | A | 545.9 / 551.8 | 10,456 / 269,705 | 28.96 / 29.15 |
| 15,922 tokens | B | 969.4 / 983.7 | 230,497 / 49,664 | 29.12 / 29.17 |

The hits and misses include the 32 decode tokens.

## Notes

- A forward of 1,024 rows copies most experts of each layer. A pass runs every layer over all of its chunks before the
  next layer, so the experts of a layer cross PCIe once a pass.
- The texts of A and B have the same MD5 for each prompt. The engine test checks a 2,088-row pass against the same
  chunks as row-major forwards, bit for bit.
- A pass keeps its hidden streams in VRAM: 335 MB for 8,192 rows.
- Strata reports 586 tokens per second at 4K and 804 at 33K on its own PC (RX 9070, MTP on). Those numbers come from
  another machine and harness.
