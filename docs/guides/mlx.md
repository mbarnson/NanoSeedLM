# The MLX loader

Each model repository on Hugging Face includes a loader file. `config.json` names it in `model_file`. An MLX app that
trusts remote code loads this file with the model. The loader supplies the seed kernels and the MLA attention. You do
not need to patch the app.

## Which loader

| Loader | Models | Source |
|---|---|---|
| `nanoseedlm_k2.py` | Seed models: GQA p4mx-q4v, MLA-P4, MLA-P3, K2-Horizon-0.9B | `tools/nanoseedlm_k2.py` in this repository |
| `k2_mla.py` | MLA models without seeds: MLA BF16, oQ4e, oQ6e, oQ8e | The model repositories |

Both loaders need the K2-Horizon model code of MLX (`mlx_lm.models.k2_horizon`). [oMLX](https://github.com/jundot/omlx)
supplies it. The mlx-lm package does not include it.

## oMLX

1. Download the model folder to the oMLX model folder or to the Hugging Face cache.
2. In oMLX, turn on **Trust Remote Code** for the model.

For K2-Horizon-0.9B, also keep ANE prefill off.

oMLX prefix reuse works with the compact MLA caches, in memory and on SSD. Read
[The KV cache](../concepts/kv-cache.md#prefix-reuse-in-omlx).

## mlx-lm

```python
from mlx_lm import load
model, tokenizer = load("PATH_TO_MODEL", trust_remote_code=True)
```

The `k2_horizon` module of oMLX must be importable.

## Long contexts (MLA models)

The loader keeps the MLA cache in 8 bits (q8) by default. If q8 cannot hold the context that you ask for, the loader
uses 4 bits (rq4). Tell the loader the longest context before you start:

- Set `NSLM_MAX_CONTEXT=524288` (tokens) in the environment of the process that loads the model, for example oMLX.
- Or add `"mla_max_context": 524288` to `config.json`.

`"mla_kv": "bf16"` in `config.json` keeps the uncompressed cache. Read [The KV cache](../concepts/kv-cache.md) for
the trade-offs and a table of the longest context for each Mac memory size.

## Recommended settings

Temperature 1.0, top_p 0.95 (IFM's defaults). The MLA models fall into a loop more often than the original in long
generations. Set a limit on `max_tokens`, or set a reasoning budget.

## Test the loader

```bash
OMLX_K2_MODEL=/path/to/omlx/patches/k2_horizon/k2_horizon_model.py make test-mlx
```

## See also

- [The model list](../reference/models.md)
- [The KV cache](../concepts/kv-cache.md)
