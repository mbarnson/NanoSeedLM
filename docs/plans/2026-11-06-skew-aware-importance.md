# Plan: skew-aware importance weights for the expert search (2026-11-06)

**Goal.** Stop the importance-weighted search from fitting a few dominant input channels at the cost of all the other
weights. Do not change the packs where the input skew is mild.

**Status.** Draft. S1 and S2 are done: `nslm-moe --h-temper`, off by default, with its unit test in `tests/test_moe.c`.
S3 waits for a CUDA or Metal machine.

## Why

- The packer weights the squared error of each expert by its imatrix h, for the seeds and for the trellis.
- On SmolLM2-135M, 1% of the MLP's input channels hold 98% of sum h. There the trellis with weights h gave 7 to 12
  times the KL of no weights, and weights h^1/2 gave the lowest KL. Read
  [Search objectives and KL](../results/2026-11-06-search-objective-kl.md).
- At a mild skew, as in most Qwen experts, the weights h gave the lowest KL. So a fixed exponent for all matrices is
  wrong in one of the two cases.
- The trellis is most at risk. One scale covers a block of 256 or 320 weights. The weighted fit of that scale follows
  the dominant channel.

## The change

`nslm-moe --h-temper off|auto|SHARE[:GAMMA]` measures, for each expert matrix, the share of sum h in its top 1% of
input channels. Above SHARE, it raises h to the power GAMMA. `auto` is 0.5:0.5. The default is `off`, so the packs do
not change. The tool prints the number of matrices that it tempered.

The count on Qwen3.8-Flash-Next, from Unsloth's imatrix with the n0 = 64 prior (S2):

| Projection | Expert matrices | Top 1% share above 0.2 | Above 0.3 | Above 0.5 (`auto`) |
|---|---|---|---|---|
| `gate_proj` | 24,576 | 0 | 0 | 0 |
| `up_proj` | 24,576 | 0 | 0 | 0 |
| `down_proj` | 24,576 | 719 | 223 | 48 |

So `auto` changes 48 of 73,728 Qwen expert matrices. The KL of the Qwen packs may not change measurably. The guard is
for models with massive activations, if the trellis or the seeds go to them.

## Steps and gates

| Step | Work | Who | Gate |
|---|---|---|---|
| S1 | `nslm_h_top_share`, `nslm_h_temper`, `nslm_h_temper_parse` in `nslm/lib_moe.c`, `--h-temper` in `nslm-moe`, and the unit test | cloud | SA |
| S2 | The count of tempered Qwen matrices for each threshold (above) | cloud (CPU) | none |
| S3 | TCQ4 (K = 4, L = 12) with `--h-temper auto` and with `0.2`, emulated into the Q8_0 shards, KL against BF16 with the saved reference | pc-cuda or HF | SB |
| S4 | If SB passes and the KL drops: TCQ3 and the K = 3 + 4 mix, then the release packs as shipped | HF | SC |

- **SA.** `test_moe` passes on macOS, Linux and Windows. With `--h-temper off`, `nslm-moe --codec tq` writes the same
  bytes as before for one layer.
- **SB.** The KL of TCQ4 with `auto` is not higher than 0.0398 by more than its standard error (about 0.0005). Report
  the KL with `0.2` too, and the number of tempered matrices.
- **SC.** The KL as shipped is not higher for any release pack. Only then can `auto` become the default.

## Hypotheses

| ID | Hypothesis | Gate |
|---|---|---|
| HS1 | `auto` tempers fewer than 1% of Qwen's expert matrices | S2 (holds: 0.07%) |
| HS2 | On Qwen, `auto` changes the KL by less than the standard error | SB |
| HS3 | A threshold of 0.2 (719 `down_proj` matrices) lowers the KL of TCQ4 by more than the standard error | SB |

## Costs

- S1 and S2: the CPU of the cloud container.
- S3: two encodes of TCQ4, about $2 each on 8 RTX PRO 6000, or about an hour each on the RTX 4080. Two KL jobs with
  the saved BF16 reference, about $2 each.
- S4: the same for each pack, if SB passes.
