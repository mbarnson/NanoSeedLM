# The model folder format

A model folder is a Hugging Face folder with safetensors shards, `config.json`, the tokenizer files and the MLX
loader. The engine reads it with `nslm/model_st.h`.

## Tensor encodings

Each tensor `NAME.weight` is stored in one of these encodings:

| Encoding | Entries |
|---|---|
| BF16 | `NAME.weight` |
| Affine Q8, Q4 (group 64, MLX layout) | `NAME.weight` (U32), `NAME.scales`, `NAME.biases` |
| Seeds, P = 4 (`SEED4P4`, 4.5 bits per weight) | `NAME.seeds` (U16), `NAME.coefs` (U16), `NAME.codes` (U8), `NAME.exp_bias` (I32) |
| Seeds, P = 3 (`SEED4`, 4.0 bits per weight) | `NAME.seeds` (U16), `NAME.nibbles` (U16: 3 int4 coefficients and the exponent code in bits 0 to 3), `NAME.exp_bias` (I32) |

Read [SeedLM](../concepts/seedlm.md) for how the engine makes weights from seeds.

## Shard layout

The writer pads each shard header to a 16 KiB page. Entries whose size is a multiple of the page come first. The
engine maps those entries into GPU buffers without a copy.

## MLA entries

An MLA model has two more keys in `config.json`:

| Key | Value |
|---|---|
| `mla_ranks` | The latent rank of each layer. A multiple of 32, at most 1024 |
| `mla_rope_dim` | 128 |

The engine reads the MLA tensors from `model-mla-delta.safetensors`. For each layer, the file has the tensors
`self_attn.mla.{kv_a_x, kv_a_v, k_rope_proj, q_rope_mix, q_lat, v_up}`. These tensors are BF16, or they use the
encoding that `nslm-mova-pack --mla` sets. The engine does not use `k_proj`. The dense layers fold `v_proj` into
`kv_a_x`.

## MLX loader keys

| Key | Use |
|---|---|
| `model_file` | The loader file: `nanoseedlm_k2.py` or `k2_mla.py` |
| `mla_max_context` | The longest context that you plan to use |
| `mla_kv` | The MLA cache format. Read [The KV cache](../concepts/kv-cache.md) |
