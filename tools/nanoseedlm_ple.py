# SPDX-License-Identifier: MIT
"""NanoSeedLM for MLX: the opt-in SSD-PLE path (mlx-vlm's qwen4_exp/ple_storage.py, 0.7.11 and 0.7.12) for an 8-bit
table too.

mlx-vlm's resident path reads a PLE table at any precision (BF16, 8-bit, 4-bit). Its SSD-PLE path, the folder view of
the cards' snippet, has layouts for 4-bit affine (group 32) and nvfp4 only: a row width of 8 values a packed word, and
each row decoded at 4 bits. The packs ship an 8-bit table (plan ple-8bit). Importing this module patches ple_storage
in place: the row width comes from the bits, and each row is decoded with its own bits and group. A 4-bit table reads
as before. The functions below follow mlx-vlm's (MIT), with the bits taken from the table's quantization.
"""

import json
import os
import struct
from pathlib import Path

import mlx.core as mx
from mlx_vlm.models.qwen4_exp import ple_storage as _ps

_ps._QUANTIZATION_LAYOUTS.setdefault((8, 32, "affine"), _ps._QUANTIZATION_LAYOUTS[(4, 32, "affine")])


def _per_word(quantization):
    return 32 // int(quantization["bits"])   # values in one packed U32


def build_quantized_ple_manifest(model_path, output_path, *, cache_rows=0):
    """mlx-vlm's manifest of the PLE byte ranges, with each table's row width from its bits."""
    model_path, output_path = Path(model_path).resolve(), Path(output_path).resolve()
    weight_map = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((model_path / "config.json").read_text())
    quant = config.get("quantization_config") or config.get("quantization", {})
    prefixes = sorted({k.rsplit(".", 1)[0] for k in weight_map if _ps.PLE_MARKER in k}, key=lambda p: int(p.rsplit(".", 1)[1]))
    if not prefixes:
        raise ValueError("checkpoint contains no Qwen4-Exp PLE shards")
    headers, shards, row_start, q0 = {}, [], 0, None

    def descriptor(key):
        f = weight_map[key]
        if f not in headers:
            with (model_path / f).open("rb") as s:
                n = struct.unpack("<Q", s.read(8))[0]
                headers[f] = (8 + n, json.loads(s.read(n)))
        start, head = headers[f]
        e = head[key]
        return {"file": f, "offset": start + int(e["data_offsets"][0]), "dtype": e["dtype"], "shape": e["shape"]}

    for p in prefixes:
        params = quant.get(p, quant)
        q = {k: params.get(k) for k in ("bits", "group_size", "mode")}
        layout = _ps._quantization_layout(q)
        if q0 is None:
            q0 = q
        elif q != q0:
            raise ValueError("all PLE shards must use the same quantization")
        t = {name: descriptor(f"{p}.{name}") for name, _ in layout}
        rows, width = int(t["weight"]["shape"][0]), int(t["weight"]["shape"][1]) * _per_word(q)
        groups = width // int(q["group_size"])
        if any(t[n]["dtype"] != d for n, d in layout) or any(t[n]["shape"] != [rows, groups] for n, _ in layout if n != "weight"):
            raise ValueError(f"PLE tensor {p!r} does not match its {q['bits']}-bit layout")
        shards.append({"row_start": row_start, "row_count": rows, **t})
        row_start += rows
    manifest = {"version": _ps.QuantizedMMapNGramEmbedding.MANIFEST_VERSION,
                "source_root": os.path.relpath(model_path, output_path.parent), "layout": "safetensors_ranges",
                "row_width": int(shards[0]["weight"]["shape"][1]) * _per_word(q0), "row_count": row_start,
                "quantization": q0, "cache_rows": int(cache_rows), "shards": shards}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class QuantizedMMapNGramEmbedding(_ps.QuantizedMMapNGramEmbedding):
    """mlx-vlm's row reader, for any bits."""

    def _init_safetensors_ranges(self, manifest, source_root):
        # mlx-vlm checks 8 values a word; seen with that many, the widths and groups it checks are the table's own
        scale = 8 // _per_word(self.quantization)
        width, group = self.row_width, self._group_size
        self.row_width, self._group_size = width * scale, group * scale
        try:
            super()._init_safetensors_ranges(manifest, source_root)
        finally:
            self.row_width, self._group_size = width, group

    def __call__(self, row_ids):
        q = self.quantization
        if q.get("mode") != "affine" or int(q["bits"]) == 4:
            return super().__call__(row_ids)
        import time
        import numpy as np
        started = time.perf_counter()
        ids = np.asarray(row_ids).astype(np.int64, copy=False)
        if ids.size and (ids.min() < 0 or ids.max() >= self.row_count):
            raise IndexError("n-gram row outside mmap table")
        flat = ids.reshape(-1)
        if not flat.size:
            return mx.zeros((*ids.shape, self.row_width), dtype=mx.bfloat16)
        unique, inverse = np.unique(flat, return_inverse=True)
        t = self._read_rows(unique)
        self._hits += flat.size - unique.size
        scales = mx.array(_ps._bfloat16_to_float32(t[1])).astype(mx.bfloat16)
        biases = mx.array(_ps._bfloat16_to_float32(t[2])).astype(mx.bfloat16)
        values = mx.dequantize(mx.array(t[0], dtype=mx.uint32), scales, biases, group_size=int(q["group_size"]), bits=int(q["bits"]))
        values = values[mx.array(inverse, dtype=mx.int64)]
        self._lookups += 1
        self._rows += ids.size
        self._elapsed += time.perf_counter() - started
        return values.reshape(*ids.shape, self.row_width)

    lookup = __call__


_ps.build_quantized_ple_manifest = build_quantized_ple_manifest   # prepare_external_ple_model calls it by this name
_ps.QuantizedMMapNGramEmbedding = QuantizedMMapNGramEmbedding     # language.py imports it when it builds the model
