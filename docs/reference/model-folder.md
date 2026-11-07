# The model folder format

A model folder is a Hugging Face folder with safetensors shards, `config.json`, the tokenizer files and the MLX
loader. The engine reads it with `nslm/model_st.h`.

## Tensor encodings

Each tensor `NAME.weight` is stored in one of these encodings:

| Encoding | Entries |
|---|---|
| BF16 | `NAME.weight` |
| Affine Q8, Q4 (group 64, MLX layout; the reader also takes Q4 group 32) | `NAME.weight` (U32), `NAME.scales`, `NAME.biases` |
| Seeds, P = 4 (`SEED4P4`, 4.5 bits per weight) | `NAME.seeds` (U16), `NAME.coefs` (U16), `NAME.codes` (U8), `NAME.exp_bias` (I32) |
| Seeds, P = 3 (`SEED4`, 4.0 bits per weight) | `NAME.seeds` (U16), `NAME.nibbles` (U16: 3 int4 coefficients and the exponent code in bits 0 to 3), `NAME.exp_bias` (I32) |

Read [SeedLM](../concepts/seedlm.md) for how the engine makes weights from seeds.

## Trellis-coded experts

A trellis pack holds its routed experts in the shards `model-tcq-NNNNN-of-NNNNN.safetensors`, one shard for each layer
and one for the MTP head (plan [tcq-safetensors](../plans/2026-11-07-tcq-safetensors.md)). Each projection is a logical
tensor `NAME.weight` of four entries. `NAME` is `language_model.model.layers.L.mlp.switch_mlp.PROJ`, or
`mtp.layers.0.mlp.switch_mlp.PROJ` for the head.

| Entry | Type | Shape | Contents |
|---|---|---|---|
| `NAME.tcq_words` | U32 | [E, rows, nw] | Each row's blocks of T weights as one bit stream, `nw = nslm_tq_row_words(cols, T, L, K)` |
| `NAME.tcq_scales` | BF16 | [E, rows, ceil(cols / T)] | One scale for each block |
| `NAME.tcq_params` | I32 | [4] | L, K, T, cols |
| `NAME.tcq_err` | F32 | [2, E] | Each expert's relative error: plain, then imatrix-weighted (square roots). The engines do not read it |

The shard's `__metadata__` also has the encode's numbers: `NAME.tcq_n0` and `NAME.tcq_seconds`.
`nslm-qwen4-pack --tq` writes a folder without the routed experts, so these entries are its only experts.

### Bank files

The encoder writes one bank file for each layer and projection, `L{layer}_{gate_proj,up_proj,down_proj}.blkt`
(`nslm-moe --codec tq`). The engines, the C reference and the MLX loader also read banks beside a folder's shards, with
`ns_add_banks`. Each bank becomes the tensor `language_model.model.layers.L.mlp.switch_mlp.PROJ.weight`, in place of
the folder's tensor of that name if it has one. A bank of layer n is the MTP head's
(`mtp.layers.0.mlp.switch_mlp.PROJ.weight`) when the folder has the routers (`mlp.gate.weight`) of layer n - 1 and of
the head, and no router for layer n. For Qwen3.8-Flash-Next, it is layer 48.

`nslm-qwen4-pack --tcq-st DIR` converts a folder's banks into `model-tcq-*` shards in place. It adds them to the index,
compares each stream with its bank, and then deletes the banks. The same folder gives the same files each time.

| Part | Contents |
|---|---|
| Header (56 bytes, `MoeBlkHeader` in `nslm/moe.h`) | magic `NSLMBLKT`, rows, cols, experts E, L (`n_seeds`), K (`n_exp`), T (`refit`), n0, the encode's seconds |
| `rel[E]`, `wrel[E]` (F32) | Each expert's relative error: plain and imatrix-weighted, square roots |
| Words (U32) | `[E][rows][nslm_tq_row_words(cols, T, L, K)]`: each row's blocks of T weights as one bit stream |
| Scales (U16, BF16) | `[E][rows][ceil(cols / T)]`: one scale for each block |

Read [Trellis-coded experts](../concepts/trellis.md) for the code and the bit stream.

## Shard layout

The writer pads each shard header to a 16 KiB page. Entries whose size is a multiple of the page come first. The
engine maps those entries into GPU buffers without a copy.

## MLA entries

An MLA model has two more keys in `config.json`:

| Key | Value |
|---|---|
| `mla_ranks` | The latent rank of each layer. A multiple of 32, at most 1024. 0 keeps the layer's GQA attention (a hybrid model) |
| `mla_rope_dim` | 128 |

The engine reads the MLA tensors from `model-mla-delta.safetensors`. For each layer, the file has the tensors
`self_attn.mla.{kv_a_x, kv_a_v, k_rope_proj, q_rope_mix, q_lat, v_up}`. These tensors are BF16, or they use the
encoding that `nslm-mova-pack --mla` sets. The engine does not use `k_proj`. The dense layers fold `v_proj` into
`kv_a_x`.

## MLX loader keys

| Key | Use |
|---|---|
| `model_file` | The loader file: `nanoseedlm_k2.py`, `nanoseedlm_qwen4.py` or `k2_mla.py`. The NanoSeedLM loaders need `nanoseedlm_seeds.py` beside them. A folder with trellis banks also needs `nanoseedlm_trellis.py` |
| `mla_max_context` | The longest context that you plan to use |
| `mla_kv` | The MLA cache format. Read [The KV cache](../concepts/kv-cache.md) |
