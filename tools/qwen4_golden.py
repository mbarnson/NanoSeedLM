# tools/qwen4_golden.py - a tiny random qwen4_exp model folder in the formats of mlx-community's 4-bit packs (Q4 group 32,
# 8-bit group 64 routers, BF16 vectors), and mlx-vlm's float32 logits for one token sequence: the C reference's golden.
#
#   python tools/qwen4_golden.py tests/data/qwen4_tiny   (the logits go to tests/data/qwen4_tiny_golden.safetensors)
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

EOS = 7
TEXT = dict(
    model_type="qwen4_exp_text", hidden_size=64, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
    head_dim=32, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
    linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=3, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, rms_norm_eps=1e-6, vocab_size=512, max_position_embeddings=4096, hc_count=4,
    hc_lowrank=32, full_attention_interval=4, ple_layer_ids=[2], ple_embed_dim=128, ple_conv_kernel_size=4,
    ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=4, seed=1234, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=16,
    indexer_compress_ratio=4, output_gate_type="sigmoid", hidden_act="silu", norm_topk_prob=True,
    eos_token_id=EOS, tie_word_embeddings=False, attention_bias=False,
    rope_parameters={"rope_type": "default", "mrope_section": [11, 11, 10], "mrope_interleaved": True,
                     "rope_theta": 10_000_000, "partial_rotary_factor": 0.25})
VISION = dict(model_type="qwen4_exp", depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
              patch_size=4, spatial_merge_size=2, temporal_patch_size=2, in_channels=3, num_position_embeddings=16,
              deepstack_visual_indexes=[])
# 48 tokens: an EOS at 19 (the n-gram context restarts) and 44 complete-block positions past the sparse budget (16).
IDS = [(37 * i + 11) % 512 for i in range(48)]
IDS[19] = EOS


def is_router(path):
    return path.endswith("mlp.gate") or path.endswith("shared_expert_gate")


def main(out):
    from mlx_vlm.models import qwen4_exp
    from mlx_vlm.utils import load_model

    cfg = dict(model_type="qwen4_exp", architectures=["Qwen4ExpForConditionalGeneration"], text_config=TEXT,
               vision_config=VISION, vocab_size=TEXT["vocab_size"], eos_token_id=EOS, tie_word_embeddings=False)
    mx.random.seed(5)
    model = qwen4_exp.Model(qwen4_exp.ModelConfig.from_dict(json.loads(json.dumps(cfg))))
    params = dict(tree_flatten(model.parameters()))
    for k, v in params.items():   # norms and gates centred on their real values, projections near their init scale
        if mx.issubdtype(v.dtype, mx.floating):
            scale = 0.3 if v.ndim == 1 else 1.0 / max(1, v.shape[-1]) ** 0.5
            params[k] = (mx.random.normal(v.shape) * scale).astype(mx.bfloat16)
    model.update(tree_unflatten(list(params.items())))
    quant = {"group_size": 32, "bits": 4, "mode": "affine"}

    def pred(path, m):
        if not hasattr(m, "to_quantized") or "vision" in path or m.weight.shape[-1] % 32:
            return False
        return {"group_size": 64, "bits": 8} if is_router(path) else True

    nn.quantize(model, group_size=32, bits=4, class_predicate=pred)
    for path, m in model.named_modules():
        if is_router(path) and isinstance(m, nn.QuantizedLinear):
            quant[path] = {"group_size": 64, "bits": 8, "mode": "affine"}
    w = {k: (v if v.dtype in (mx.uint32, mx.int64) else v.astype(mx.bfloat16)) for k, v in tree_flatten(model.parameters())}
    os.makedirs(out, exist_ok=True)
    mx.save_safetensors(f"{out}/model.safetensors", w, metadata={"format": "mlx"})
    cfg["quantization"] = quant
    cfg["quantization_config"] = quant
    with open(f"{out}/config.json", "w") as f:
        json.dump(cfg, f, indent=1)

    model = load_model(__import__("pathlib").Path(out), strict=True)
    model.update(tree_map(lambda v: v.astype(mx.float32) if v.dtype == mx.bfloat16 else v, model.parameters()))
    from mlx_vlm.models.qwen4_exp import language
    rows, call = [], language.ShardedEmbedding.__call__   # capture the PLE n-gram rows on their way to the table
    language.ShardedEmbedding.__call__ = lambda self, idx: (rows.append(idx), call(self, idx))[1]
    logits = model.language_model(mx.array([IDS])).logits[0].astype(mx.float32)
    language.ShardedEmbedding.__call__ = call
    mx.save_safetensors(f"{out}_golden.safetensors",
                        {"ids": mx.array(IDS, dtype=mx.int32), "logits": logits, "ple_rows": rows[0][0].astype(mx.int64)})
    print(f"{out}: {len(w)} tensors, logits {logits.shape}, argmax {mx.argmax(logits, axis=-1).tolist()[:12]} ...")


if __name__ == "__main__":
    main(sys.argv[1])
