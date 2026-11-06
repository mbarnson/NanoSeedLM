"""tools/st_pad.py FILE... - rewrite each safetensors file so every tensor is aligned for its type: the header padded with
spaces to 8 bytes, as the Rust writer does, and the tensors in order of element size (8, 4, 2, 1 bytes), so none needs a
gap. mx.save_safetensors does neither, and the C readers then load misaligned values (UBSan)."""
import json
import struct
import sys

SIZE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2}


def pad8(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n])
    meta = head.pop("__metadata__", None)
    body = raw[8 + n:]
    out, at = {}, 0
    if meta is not None:
        out["__metadata__"] = meta
    parts = []
    for k in sorted(head, key=lambda k: (-SIZE.get(head[k]["dtype"], 1), k)):
        a, b = head[k]["data_offsets"]
        out[k] = dict(head[k], data_offsets=[at, at + b - a])
        parts.append(body[a:b])
        at += b - a
    h = json.dumps(out, separators=(",", ":")).encode()
    h += b" " * (-(8 + len(h)) % 8)
    new = struct.pack("<Q", len(h)) + h + b"".join(parts)
    if new == raw:
        return False
    open(path, "wb").write(new)
    return True


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, "rewritten" if pad8(p) else "already aligned")
