#!/usr/bin/env python3
"""tools/page_resident.py - how much of a qwen4 pack's files sits in the page cache (macOS or Linux, stdlib only): mincore()
over each tensor's byte range of the folder's *.safetensors shards, by kind, and over its *.blkt trellis banks.

  python3 tools/page_resident.py PACK_DIR [--drop]

Kinds: the PLE table (".ple." tensors), the routed experts in the shards ("switch_mlp", which a pack's .blkt banks replace,
so then they are not read), the MTP head ("mtp"), the other tensors, and the .blkt banks. A resident page (16 KB on Apple
silicon) counts its overlap with each tensor, so small tensors that share a page are not counted twice.

--drop first evicts every safetensors range (macOS: msync(MS_INVALIDATE) on a read-only mapping; Linux:
posix_fadvise(POSIX_FADV_DONTNEED); no root needed; the banks stay). Then a run shows what it reads: drop, run the engine,
then run this tool again.
"""
import ctypes, ctypes.util, glob, json, os, struct, sys

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
PAGE = os.sysconf("SC_PAGE_SIZE")
PROT_READ, MAP_SHARED = 1, 1
MS_INVALIDATE = 2   # <sys/mman.h>, macOS (Linux's msync does not evict; posix_fadvise does there)
KINDS = ["PLE table", "routed experts in the shards", "MTP head", "other tensors", ".blkt banks"]


def resident(path, ranges, drop=False):
    """Bytes of the whole pages under ranges [(a, b), ...] of path that are in the page cache (after msync if drop)."""
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    if drop and hasattr(os, "posix_fadvise"):   # Linux
        for a, b in ranges:
            os.posix_fadvise(fd, a, b - a, os.POSIX_FADV_DONTNEED)
    p = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0) if size else None
    os.close(fd)
    if not size:
        return 0
    if p is None or p == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), "mmap " + path)
    tot = 0
    try:
        for a, b in ranges:
            a0, b1 = a // PAGE * PAGE, min(size, (b + PAGE - 1) // PAGE * PAGE)
            if b1 <= a0:
                continue
            if drop and not hasattr(os, "posix_fadvise"):   # macOS
                libc.msync(p + a0, b1 - a0, MS_INVALIDATE)
            vec = ctypes.create_string_buffer((b1 - a0 + PAGE - 1) // PAGE)
            if libc.mincore(p + a0, b1 - a0, vec):
                raise OSError(ctypes.get_errno(), "mincore " + path)
            for i, c in enumerate(vec.raw):   # a resident page counts its overlap with the range, so pages shared by
                if c & 1:                     # small tensors are not counted twice
                    ps = a0 + i * PAGE
                    tot += min(ps + PAGE, b) - max(ps, a)
    finally:
        libc.munmap(p, size)
    return tot


def kind_of(name):
    if ".ple." in name:
        return KINDS[0]
    if "mtp" in name:
        return KINDS[2]
    if "switch_mlp" in name:
        return KINDS[1]
    return KINDS[3]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1 or not os.path.isdir(args[0]):
        sys.exit(__doc__)
    d, drop = args[0], "--drop" in sys.argv[1:]
    res, tot = dict.fromkeys(KINDS, 0), dict.fromkeys(KINDS, 0)
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        ranges = {k: [] for k in KINDS[:4]}
        for name, v in hdr.items():
            if name != "__metadata__":
                ranges[kind_of(name)].append((8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1]))
        for k, r in ranges.items():
            res[k] += resident(f, r, drop)
            tot[k] += sum(b - a for a, b in r)
    for f in sorted(glob.glob(os.path.join(d, "*.blkt"))):
        s = os.path.getsize(f)
        res[KINDS[4]] += resident(f, [(0, s)])
        tot[KINDS[4]] += s
    if drop:
        how = "posix_fadvise DONTNEED" if hasattr(os, "posix_fadvise") else "msync MS_INVALIDATE"
        print("dropped every safetensors range (%s); now resident:" % how)
    for k in KINDS:
        if tot[k]:
            u, s = ("GB", 1e9) if tot[k] >= 1e9 else ("MB", 1e6)
            print("%-30s %8.2f %s resident of %8.2f %s" % (k, res[k] / s, u, tot[k] / s, u))


if __name__ == "__main__":
    main()
