#!/usr/bin/env python3
"""tools/gguf_experts.py - the routed experts of one layer of a Qwen3.8-Flash-Next GGUF (e.g. an Unsloth UD pack),
dequantized to BF16 safetensors ('w' [experts][rows][cols]), one file per projection, for nslm-moe --probe-ref. Prints
each tensor's GGUF type and bits per weight.

  python3 tools/gguf_experts.py --layer 20 --out DIR FILE.gguf [FILE.gguf ...]
"""
import argparse
import json
import os
import struct

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

NAMES = {"gate_proj": "ffn_gate_exps", "up_proj": "ffn_up_exps", "down_proj": "ffn_down_exps"}


def bf16(a):
    """float32 -> BF16 bits, round to nearest even."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def write_st(path, bits, shape):
    head = json.dumps({"w": {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, bits.nbytes]}}).encode()
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        f.write(bits.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("gguf", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    want = {"blk.%d.%s.weight" % (a.layer, g): p for p, g in NAMES.items()}
    for path in a.gguf:
        for t in GGUFReader(path).tensors:
            if t.name not in want:
                continue
            shape = tuple(int(n) for n in reversed(t.shape))   # GGUF ne0 is the column count
            w = dequantize(t.data, t.tensor_type).reshape(shape)
            write_st(os.path.join(a.out, "L%d_%s.safetensors" % (a.layer, want[t.name])), bf16(w), shape)
            print("%s: %s, %.4f bits per weight, %s" % (t.name, t.tensor_type.name, t.n_bytes * 8 / t.n_elements, shape))


if __name__ == "__main__":
    main()
