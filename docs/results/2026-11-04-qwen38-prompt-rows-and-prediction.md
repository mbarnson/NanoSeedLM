# Prompt rows of 1,024 and a decode expert prediction for Qwen3.8-Flash-Next on CUDA (2026-11-04)

**Result.** The test used an RTX PRO 6000 with an 8 GB expert cache. A prompt of 3,703 tokens of P4 runs at 541
tokens per second in forwards of 1,024 rows. In forwards of 128 rows, it runs at 151. The cache misses go from 274k to
64k.
A prediction of the next layer's experts in decode cut the misses by 44%. But decode went from 40.6 to 39.0 tokens per
second, so we removed it.

**Status.** Current. The test used HF Jobs, not the RTX 4080. The RTX 4080 has PCIe 4, so a cache miss costs more
there, and the prediction may help on it.

## Setup

- Machine: HF Jobs `rtx-pro-6000` (RTX PRO 6000 Blackwell, 96 GB). The load average was 5.3 to 11.0 in all runs.
- Models: [Qwen3.8-Flash-Next-NanoSeedLM-P4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4) for the
  prompt rows, [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) for
  the prediction.
- Prompt rows: the first 16,000 characters of the wikitext-2 test set (3,703 tokens), raw, then 32 or 48 greedy tokens.
  The source of `15b4fab`, with `Q4E_MAX_ROWS` set to 128, 512 or 1,024. Jobs `6aebc7b90e0338d1bb804990` and, after the
  QSA order fix of `6fef57c`, `6aebcbe3c78863f9c6330e3b`.
- Prediction: "Why do tide pools matter? Answer in three sentences." with the chat template, 161 greedy tokens. Commit
  `1cd7c57`, with and without `NSLM_NO_PREDICT=1`. Job `6aebc860c78863f9c6330c64`.

```bash
nslm-qwen4-gen --model PACK --prompt "$(cat prompt.txt)" --raw --max-tokens 32 --ctx 8192 --expert-cache-gb 8
```

## Results

The 3,703-token prompt of P4, two runs at each row count:

| Rows a forward | Prompt, tokens/s | Cache misses | Router and cache, ms | Experts, ms |
|---|---|---|---|---|
| 128 | 150.0 / 151.6 | 274,302 / 274,108 | 15,271 / 15,301 | 4,244 / 4,247 |
| 512 | 358.1 / 361.6 | 112,538 / 112,915 | 6,275 / 6,298 | 1,782 / 1,783 |
| 1,024 | 539.0 / 542.5 | 64,404 / 69,329 | 3,582 / 3,875 | 954 / 1,203 |
| 128, QSA in block order | 150.7 / 152.6 | | | |
| 1,024, QSA in block order | 530.7 / 540.2 | | | |

The miss and stage counts include the 32 decode tokens, except in the first run of 1,024 rows (0 tokens).

P3 decode of 161 tokens, two runs each:

| Prediction | Decode, tokens/s | Hits / misses | Router and cache, ms | Experts, ms | Hyper-connections, ms |
|---|---|---|---|---|---|
| On (`1cd7c57`) | 38.91 / 39.01 | 67,461 / 14,396 | 1,220 / 1,218 | 1,250 / 1,246 | 992 / 996 |
| Off | 40.56 / 40.58 | 56,223 / 25,634 | 1,366 / 1,364 | 1,131 / 1,133 | 882 / 878 |

## Notes

- **Prompt rows.** Each forward copies the missing experts of each layer once. Forwards of 1,024 rows copy each expert
  fewer times. Forwards with logits keep 128 rows, so the logits buffer stays at 128 x vocab floats. The cost is VRAM:
  the activations of 1,024 rows take about 0.7 GB (an estimate from the buffer sizes, not measured). The expert cache
  does not get that VRAM.
- **Prediction.** The next layer's router ran on this layer's state, and a second stream copied the experts that it
  chose. The copies and the extra router slowed the other stages by 8% to 13%, more than the miss time that they saved.
  PCIe 5 makes a miss cheap on this GPU. Commit `1cd7c57` keeps the code for a test on the RTX 4080.
- **The empty run.** The first run of 1,024 rows stopped at the first token. QSA wrote its selected blocks in the order
  of an atomic counter, and attention sums the keys in that order. So the logits differed in their last bits from run to
  run past position 2,048, and a near tie can change a greedy token. `6fef57c` writes the blocks in block order. After
  it, the four texts of 128 and 1,024 rows were the same (MD5 `2411721d`).
- Strata's own numbers, on another PC (RX 9070, ISTA IQ3_XXS, MTP on), are 586 to 833 tokens per second for prompts.
  Plan step S0.3 measures Strata on the RTX 4080. Until then, the two sets of numbers are not comparable.
