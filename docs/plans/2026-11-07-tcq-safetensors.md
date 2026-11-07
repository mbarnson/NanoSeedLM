# Plan: trellis experts in safetensors shards (2026-11-07)

**Goal.** The three trellis packs (TCQ4, TCQ3-D4, TCQ3) carry their routed experts in safetensors shards, not in 147
`L{layer}_{proj}.blkt` bank files. The bytes of the codes and the scales do not change, so the engines and the MLX
loader give the same tokens. Standard tools can read every file of a pack.

**Status.** In progress. TS1 to TS4 pass on the M4 Max (TA, TC). TS5 runs as an HF job.

## Why

- Matt (2026-11-07): the packs must use `.safetensors`, not a format of our own.
- A `.blkt` file is the encoder's work file (`nslm-moe --codec tq`): a 56-byte header, two error arrays, the code
  words and the scales. The packs shipped these files as they were. No decision made them a distribution format.
- The words of a `.blkt` bank start at byte 56 + 8 E, not on a page. In a shard they start on a 16 KiB page, so the
  Metal engine can map them without a copy, as it maps the other shards.
- A bank stream has the offset 0 and the shard 0, so `ns_mappable` tests it against shard 0. In a shard, the stream
  has its true offset and shard.

## The format

Each projection's bank becomes four entries. `NAME` is `language_model.model.layers.L.mlp.switch_mlp.PROJ`, or
`mtp.layers.0.mlp.switch_mlp.PROJ` for the MTP head (the head's bank was layer 48).

| Entry | Type | Shape | Contents |
|---|---|---|---|
| `NAME.tcq_words` | U32 | [E, rows, nw] | The code words of each row, `nw = nslm_tq_row_words(cols, T, L, K)` |
| `NAME.tcq_scales` | BF16 | [E, rows, ceil(cols / T)] | One scale for each block of T weights |
| `NAME.tcq_params` | I32 | [4] | L, K, T, cols |
| `NAME.tcq_err` | F32 | [2, E] | Each expert's relative error: plain, then imatrix-weighted |

- A layer's three projections go in one shard: `model-tcq-00001-of-00049.safetensors` (48 layers and the head).
- The writer pads the header to 16 KiB and puts the page-sized entries first, as `ns_write_raw` does.
- The index lists the new shards. The other shards do not change, so a download keeps them.
- The header's encode statistics (n0, seconds) go into the shard's `__metadata__`.
- The engines and the MLX loader still read `.blkt` banks beside a folder's shards, for local work.

## Steps and gates

| Step | Work | Gate |
|---|---|---|
| TS1 | `ns_open` reads the `tcq_*` entries as `NS_TCQ`. A C test writes one bank in both forms | TA |
| TS2 | `nslm-qwen4-pack --tq` writes the banks as shards. `nslm-qwen4-pack --tcq-st DIR` converts a pack folder in place | TA, TB |
| TS3 | The MLX loader (`nanoseedlm_trellis.py`) reads the entries. `test_nanoseedlm_trellis.py` covers both forms | TA |
| TS4 | M4 Max: a copy of TCQ3, converted. The Metal engine and the MLX loader against the `.blkt` folder | TC |
| TS5 | An HF job converts the three repositories: one commit each, with the loaders and cards. The history stays | TB, TD |
| TS6 | The start scripts pin the new revisions. `VERSION` 0.1.2, so CI makes a release draft. Docs and a result | TD |
| TS7 | CUDA: the converted TCQ3 against the `.blkt` TCQ3 on an HF GPU job | TE |

- **TA.** The streams of each bank are byte for byte the same in both forms. The C reference, the test of each engine
  and the loader test pass.
- **TB.** The conversion is deterministic: the same folder gives shards with the same SHA-256 twice. Each shard on HF
  has the SHA-256 of the M4 Max conversion of the same pack.
- **TC.** On the M4 Max, the converted folder gives the same token IDs as the `.blkt` folder: 2 prompts, 128 tokens,
  temperature 0, in `nslm-qwen4-gen` and in the MLX loader. Decode speed is within 2%. The peak memory is not higher.
- **TD.** `start-here.sh` at the new pins downloads and checks each file. A folder at the old pins keeps its
  unchanged shards, and downloads only the new ones.
- **TE.** The CUDA engine gives the same token IDs for the two folders.

## Hypotheses

- H1: the change is lossless, so the token IDs are equal (not only close).
- H2: on Metal the banks map without a copy. The load is faster, and the anonymous memory falls by about the banks'
  size (47.2 GB for TCQ3).
- H3: Xet stores the same bytes once, so the upload of the new shards is small.

## Costs

- TS5: `cpu-xl` ($1.00/h), about 2 hours for the three packs: about $2. Matt allows more than the weekly cap for
  this work (2026-11-07).
- TS7: `rtx-pro-6000x2` ($5.50/h), about 30 minutes: about $3.
- A download of a pack at the old pins downloads its banks again: about 47 GB for each pack.
- The v0.1.1 engines do not read the new revisions. The old revisions stay on HF, so the v0.1.1 start scripts still
  work. A release folder needs v0.1.2 for the new revisions.

## Risks

- The MLX loader's strict load fails on an entry it does not consume. The loader test covers the strict load.
- A checkout without a compiler falls back to the release engine. It cannot read the new packs until v0.1.2 is
  published.

## See also

- [Trellis-coded experts](../concepts/trellis.md)
- [The model folder format](../reference/model-folder.md)
- [Plan ple-8bit](2026-11-06-ple-8bit.md), which added shards to the packs in the same way
