# imat_skew.py - skew of the importance h that the Qwen trellis encode used: Unsloth's imatrix, per expert, blended
# with the layer mean by n0 = 64 (nslm_imatrix_h).  Share of sum h in the top 1% of input channels, and max / median.
import struct, sys, numpy as np
path = sys.argv[1]
f = open(path, "rb"); buf = memoryview(np.memmap(path, dtype=np.uint8, mode="r"))
pos = 0
def rd(fmt):
    global pos
    v = struct.unpack_from(fmt, buf, pos); pos += struct.calcsize(fmt); return v[0] if len(v) == 1 else v
def rstr():
    global pos
    n = rd("<Q"); s = bytes(buf[pos:pos + n]).decode(); pos += n; return s
SZ = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        et, n = rd("<I"), rd("<Q"); return [rval(et) for _ in range(n)]
    return rd(SZ[t])
assert bytes(buf[:4]) == b"GGUF"; pos = 4
ver, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
kv = {}
for _ in range(nkv):
    k = rstr(); kv[k] = rval(rd("<I"))
info = {}
for _ in range(nt):
    name = rstr(); nd = rd("<I"); dims = [rd("<Q") for _ in range(nd)]; t = rd("<I"); off = rd("<Q")
    info[name] = (dims, t, off)
al = kv.get("general.alignment", 32); base = (pos + al - 1) // al * al
def tensor(name):
    dims, t, off = info[name]; assert t == 0
    n = int(np.prod(dims)); return np.frombuffer(buf[base + off: base + off + 4 * n], dtype=np.float32).reshape(dims[::-1])
n0 = 64.0
print(f"{len(info)} tensors; e.g. {sorted(info)[:2]}")
print("SmolLM2 layer 15 MLP input, for comparison: top 1% 98.2%, max/median 86458")
for l in (0, 20, 40):
    for proj in ("gate", "up", "down"):
        s = tensor(f"blk.{l}.ffn_{proj}_exps.weight.in_sum2"); c = tensor(f"blk.{l}.ffn_{proj}_exps.weight.counts").reshape(-1)
        E, cols = s.shape
        prior = s.sum(0) / max(c.sum(), 1)
        top, ratio = [], []
        for e in range(E):
            h = (s[e] + n0 * prior) / (c[e] + n0)
            hs = np.sort(h)[::-1]
            top.append(hs[: max(1, cols // 100)].sum() / hs.sum()); ratio.append(hs[0] / np.median(hs))
        top, ratio = np.array(top), np.array(ratio)
        print(f"layer {l:2d} {proj:4s} ({cols} inputs): top 1% share median {np.median(top):.1%} (max {top.max():.1%}); "
              f"max/median median {np.median(ratio):.0f} (max {ratio.max():.0f})")
