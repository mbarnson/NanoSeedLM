#!/usr/bin/env python3
"""tools/gguf_expert_bytes.py - the routed experts' bytes and bits for each weight of a GGUF pack on the Hub, from the
shards' headers only (HTTP range reads). Each tensor's size comes from the data offsets, so new types count too.

  python3 tools/gguf_expert_bytes.py REPO REVISION FOLDER     (e.g. unsloth/Qwen3.8-Flash-Next-GGUF d962012... UD-IQ3_XXS)
"""
import struct, sys
from huggingface_hub import HfFileSystem
repo, rev, folder = sys.argv[1], sys.argv[2], sys.argv[3]
fs = HfFileSystem()
files = sorted(f for f in fs.ls(f"{repo}@{rev}/{folder}", detail=False) if f.endswith(".gguf"))
def rd(f, n): b = f.read(n); assert len(b) == n; return b
def u32(f): return struct.unpack("<I", rd(f, 4))[0]
def u64(f): return struct.unpack("<Q", rd(f, 8))[0]
def s(f): return rd(f, u64(f)).decode()
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(f, t):
    if t == 8: return s(f)
    if t == 9:
        et, n = u32(f), u64(f)
        return [val(f, et) for _ in range(n)]
    return rd(f, SZ[t])
eb = ew = 0
by = {}
for path in files:
    size = fs.info(path)["size"]
    with fs.open(path, "rb", block_size=1 << 22) as f:
        assert rd(f, 4) == b"GGUF"; u32(f)
        nt, nk = u64(f), u64(f)
        align = 32
        for _ in range(nk):
            k = s(f); t = u32(f); v = val(f, t)
            if k == "general.alignment": align = struct.unpack("<I", v)[0]
        ts = []
        for _ in range(nt):
            name = s(f); nd = u32(f); dims = [u64(f) for _ in range(nd)]; ty = u32(f); off = u64(f)
            n = 1
            for d in dims: n *= d
            ts.append((off, name, n, ty))
        start = (f.tell() + align - 1) // align * align
    ts.sort()
    for i, (off, name, n, ty) in enumerate(ts):
        nxt = ts[i + 1][0] if i + 1 < len(ts) else size - start
        if "_exps" in name and ".ffn_" in name and name.startswith("blk."):
            b = nxt - off
            eb += b; ew += n
            by.setdefault(ty, [0, 0]); by[ty][0] += b; by[ty][1] += n
print(f"{folder}: {eb / 1e9:.2f} GB, {8 * eb / ew:.3f} bits per expert weight; by type: " +
      ", ".join(f"{t}: {8 * b / n:.3f} bits" for t, (b, n) in sorted(by.items())))
