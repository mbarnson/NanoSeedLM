#!/usr/bin/env python3
"""tools/nanoseedlm_ple.py: the opt-in SSD-PLE path (ple_storage.py) reads an 8-bit table row for row as its tensors
give it, and a 4-bit table as unpatched mlx-vlm does. It also makes the view (prepare_external_ple_model) of a pack with the 8-bit table.

  python tools/test_nanoseedlm_ple.py      (needs mlx and mlx-vlm with qwen4_exp/ple_storage.py)
"""
import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlx_vlm.models.qwen4_exp import ple_storage as ps  # noqa: E402

ORIG_BUILD, ORIG_CLASS = ps.build_quantized_ple_manifest, ps.QuantizedMMapNGramEmbedding
import nanoseedlm_ple  # noqa: E402,F401  (patches ps)

TMP = Path(tempfile.mkdtemp(prefix="nslm_ple_"))
NAME = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{}"
SHARDS, ROWS, WIDTH = 3, 200, 160


def folder(bits):
    """A folder with a PLE table of SHARDS shards at this many bits, group 32, as the packs store it."""
    d = TMP / f"q{bits}"
    d.mkdir()
    rng = np.random.default_rng(bits)
    w, entries = {}, {}
    for k in range(SHARDS):
        x = mx.array(rng.normal(0, 0.01, (ROWS, WIDTH)).astype(np.float32)).astype(mx.bfloat16)
        q, s, b = mx.quantize(x, group_size=32, bits=bits)
        w.update({NAME.format(k) + ".weight": q, NAME.format(k) + ".scales": s, NAME.format(k) + ".biases": b})
        entries[NAME.format(k)] = {"group_size": 32, "bits": bits, "mode": "affine"}
    mx.save_safetensors(str(d / "model-00001-of-00001.safetensors"), w)
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in w}}))
    (d / "config.json").write_text(json.dumps({"text_config": {}, "quantization": {"group_size": 64, "bits": 4, **entries}}))
    return d, w


def direct(w, bits, ids):
    out = []
    for r in ids.reshape(-1).tolist():
        k, i = divmod(r, ROWS)
        n = NAME.format(k)
        out.append(mx.dequantize(w[n + ".weight"][i:i + 1], w[n + ".scales"][i:i + 1], w[n + ".biases"][i:i + 1], group_size=32, bits=bits))
    return np.array(mx.concatenate(out).astype(mx.float32)).reshape(*ids.shape, WIDTH)


ids = np.array([[0, 7, ROWS - 1], [ROWS, 2 * ROWS + 5, SHARDS * ROWS - 1]])
bad = 0
for bits in (8, 4):
    d, w = folder(bits)
    man = ps.build_quantized_ple_manifest(d, d / "ple-store.json")
    got = np.array(ps.QuantizedMMapNGramEmbedding(d / "ple-store.json")(ids).astype(mx.float32))
    diff = float(np.abs(got - direct(w, bits, ids)).max())
    ok = man["row_width"] == WIDTH and diff == 0
    print(f"{bits}-bit table: row width {man['row_width']}, rows against their tensors: max difference {diff}: {'ok' if ok else 'FAIL'}")
    bad += not ok
    if bits == 4:   # as unpatched mlx-vlm reads it
        ORIG_BUILD(d, d / "orig.json")
        ref = np.array(ORIG_CLASS(d / "orig.json")(ids).astype(mx.float32))
        same = bool(np.array_equal(got, ref))
        print(f"4-bit table: the same rows as unpatched mlx-vlm: {'ok' if same else 'FAIL'}")
        bad += not same
    else:
        ps.prepare_external_ple_model(d, TMP / "q8-view")
        v = json.loads((TMP / "q8-view" / "ple-store.json").read_text())
        ok = v["quantization"]["bits"] == 8 and v["row_width"] == WIDTH
        print(f"8-bit view: prepare_external_ple_model: {'ok' if ok else 'FAIL'}")
        bad += not ok
print("test_nanoseedlm_ple:", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
