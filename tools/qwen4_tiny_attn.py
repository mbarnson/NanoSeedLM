# tools/qwen4_tiny_attn.py SRC DST - qwen4_tiny with strong attention (q/k norms 1 + w times NORM, v and o times VO),
# so engine tests see attention bugs. Pure Python.
import json
import os
import shutil
import struct
import sys

NORM, VO = 3.0, 4.0


def bf16_to_f32(b):
    return [struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in struct.unpack("<%dH" % (len(b) // 2), b)]


def f32_to_bf16(v):
    out = []
    for x in v:
        u = struct.unpack("<I", struct.pack("<f", x))[0]
        out.append(((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF)   # nearest, ties to even
    return struct.pack("<%dH" % len(out), *out)


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src, "config.json"), os.path.join(dst, "config.json"))
    raw = open(os.path.join(src, "model.safetensors"), "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n])
    body = bytearray(raw[8 + n:])
    changed = 0
    for name, t in head.items():
        if name == "__metadata__" or ".self_attn." not in name or t["dtype"] != "BF16":
            continue
        a, b = t["data_offsets"]
        v = bf16_to_f32(bytes(body[a:b]))
        if name.endswith(("q_norm.weight", "k_norm.weight")):
            v = [NORM * (1.0 + x) - 1.0 for x in v]
        elif any(name.endswith(p + "_proj." + s) for p in ("v", "o") for s in ("scales", "biases")):
            v = [VO * x for x in v]
        else:
            continue
        body[a:b] = f32_to_bf16(v)
        changed += 1
    with open(os.path.join(dst, "model.safetensors"), "wb") as f:
        f.write(raw[:8 + n])
        f.write(body)
    print("%s: %d attention tensors scaled" % (dst, changed))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
