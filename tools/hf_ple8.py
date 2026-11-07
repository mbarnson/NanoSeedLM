#!/usr/bin/env python3
"""tools/hf_ple8.py - plan ple-8bit P2-P4: Qwen's BF16 PLE n-gram table in MLX's 8-bit form (group 32), and a pull request on
each pack that puts it in place of the pack's 4-bit table. One file at a time, so cpu-upgrade's 50 GB of disk is enough.

  python3 tools/hf_ple8.py table OUT            the 8-bit table in OUT: one file for each BF16 file that holds shards
  python3 tools/hf_ple8.py pack REPO OUT        a pull request on REPO (not merged): the 4-bit table out, OUT's files in

As an HF job, by the account that owns the packs (its token a job secret):
  hf jobs run --flavor cpu-upgrade --timeout 6h --secrets HF_TOKEN -e HF_XET_CHUNK_CACHE_SIZE_BYTES=0 -v hf://buckets/txgsync/nslm-qwen38:/b python:3.12 \
      bash -c 'pip install -q numpy "huggingface_hub[hf_xet]" && S=/b/src-REV/tools/hf_ple8.py &&
               python3 $S table /b/ple8 && for P in TCQ4 TCQ3-D4 TCQ3 P4 P3; do
               python3 $S pack txgsync/Qwen3.8-Flash-Next-NanoSeedLM-$P /b/ple8; done'
"""
import json
import os
import re
import struct
import subprocess
import sys

SRC, SRC_REV = "Qwen/Qwen3.8-Flash-Next", "7ecef55a2e549814c62143fcf8b5be457695e3b9"
BF16 = re.compile(r"model\.language_model\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
OLD = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shards\.\d+\.(weight|scales|biases)$")
Q8 = {"group_size": 32, "bits": 8, "mode": "affine"}   # 64 does not divide the table's 160 columns
WORK = os.environ.get("WORK", "/w")
# The table's big files go to and from the bucket by its API. The bucket mount keeps a local copy of each file written
# through it: 28 of the 33 filled cpu-upgrade's 50 GB of disk (job 6aee71f5). The mount serves only small reads.
URI = os.environ.get("PLE8_URI", "hf://buckets/txgsync/nslm-qwen38/ple8")


def hf_cp(src, dst):
    subprocess.run(["hf", "cp", src, dst], check=True, stdout=subprocess.DEVNULL)
SIZE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2}


def st_read(raw):
    """A safetensors file's header and data section."""
    n = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8:8 + n]), raw[8 + n:]


def st_write(head, parts):
    """A safetensors file of these tensors {name: (entry, bytes)}: each aligned for its type, as tools/st_pad.py."""
    out, body, at = ({"__metadata__": head} if head else {}), [], 0
    for k in sorted(parts, key=lambda k: (-SIZE.get(parts[k][0]["dtype"], 1), k)):
        e, b = parts[k]
        out[k] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [at, at + len(b)]}
        body.append(b)
        at += len(b)
    h = json.dumps(out, separators=(",", ":")).encode()
    h += b" " * (-(8 + len(h)) % 8)
    return struct.pack("<Q", len(h)) + h + b"".join(body)


def without(raw, drop):
    """The file's bytes without the tensors that drop(name) selects, or None when none would be left."""
    head, body = st_read(raw)
    meta = head.pop("__metadata__", None)
    keep = {k: (e, body[e["data_offsets"][0]:e["data_offsets"][1]]) for k, e in head.items() if not drop(k)}
    return st_write(meta, keep) if keep else None


def pack_name(bf16_name):
    """Qwen's tensor name to the packs' module name: ...layers.L.ple...shard_N.weight to ...shards.N"""
    m = BF16.match(bf16_name)
    return m and f"language_model.model.layers.{m.group(1)}.ple.ple_embedding.ngram_embedding.shards.{m.group(2)}"


def new_config(cfg, modules):
    """config.json with the 8-bit entry for each module, in every quantization block it has."""
    for blk in (cfg, cfg.get("text_config") or {}):
        for key in ("quantization", "quantization_config"):
            if isinstance(blk.get(key), dict):
                for m in modules:
                    blk[key][m] = dict(Q8)
    return cfg


def new_index(idx, ple8, total):
    """The index without the 4-bit table, with ple8 {tensor: file} in its place, and the new total size in bytes."""
    wm = {k: f for k, f in idx["weight_map"].items() if not OLD.search(k)}
    wm.update(ple8)
    return {"metadata": dict(idx.get("metadata", {}), total_size=total), "weight_map": dict(sorted(wm.items()))}


def bf16(x):
    """f32 to BF16 bits, rounded to nearest even."""
    import numpy as np
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def encode8(w):
    """w (f32, rows by cols): MLX's 8-bit affine form, group 32, as (codes U32, scales, biases BF16 bits). The codes are
    taken against the stored BF16 scale and bias. mx.quantize takes them against the unrounded ones: on 10,000 rows of
    this table it gives 1.19 times the RMS error of Unsloth's Q8_0 table, and this gives 0.88 times."""
    import numpy as np
    f32 = lambda h: (h.astype(np.uint32) << 16).view(np.float32)
    g = w.reshape(-1, 32)
    lo, hi = g.min(1), g.max(1)
    sb, bb = bf16(np.maximum((hi - lo) / 255, 1e-12)), bf16(lo)
    q = np.clip(np.rint((g - f32(bb)[:, None]) / f32(sb)[:, None]), 0, 255).astype(np.uint8)
    rows = w.shape[0]
    return q.reshape(rows, -1).view(np.uint32), sb.reshape(rows, -1), bb.reshape(rows, -1)


def quantize(raw, chunk=250000):
    """One BF16 file's n-gram shards in 8 bits: {tensor: (entry, bytes)}, and the RMS error against BF16."""
    import numpy as np
    head, body = st_read(raw)
    out, se, n = {}, 0.0, 0
    for k, e in head.items():
        name = k != "__metadata__" and pack_name(k)
        if not name:
            continue
        a, b = e["data_offsets"]
        rows, cols = e["shape"]
        h = np.frombuffer(body[a:b], dtype=np.uint16).reshape(rows, cols)
        parts = []
        for r in range(0, rows, chunk):   # a chunk at a time: about 1 GB of f32
            w = (h[r:r + chunk].astype(np.uint32) << 16).view(np.float32)
            q, sb, bb = encode8(w)
            d = (q.view(np.uint8).reshape(-1, 32) * (sb.reshape(-1, 1).astype(np.uint32) << 16).view(np.float32)
                 + (bb.reshape(-1, 1).astype(np.uint32) << 16).view(np.float32))
            se += float(((d.reshape(w.shape) - w) ** 2).sum())
            n += w.size
            parts.append((q, sb, bb))
        for i, (suf, dt) in enumerate(((".weight", "U32"), (".scales", "BF16"), (".biases", "BF16"))):
            t = np.concatenate([p[i] for p in parts])
            out[name + suf] = ({"dtype": dt, "shape": list(t.shape)}, t.tobytes())
    return out, (se / max(n, 1)) ** 0.5


def table(out_dir):
    from huggingface_hub import hf_hub_download
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    idx = json.load(open(hf_hub_download(SRC, "model.safetensors.index.json", revision=SRC_REV)))
    files = sorted({f for k, f in idx["weight_map"].items() if BF16.match(k)})
    wmap, nbytes = {}, 0
    for i, f in enumerate(files, 1):
        name = f"model-ple8-{i:05d}-of-{len(files):05d}.safetensors"
        dst = os.path.join(out_dir, name)
        if not os.path.exists(dst):
            p = hf_hub_download(SRC, f, revision=SRC_REV, local_dir=os.path.join(WORK, "src"))
            parts, rms = quantize(open(p, "rb").read())
            os.remove(p)
            tmp = os.path.join(WORK, name)
            open(tmp, "wb").write(st_write({"format": "mlx"}, parts))
            hf_cp(tmp, f"{URI}/{name}")
            os.remove(tmp)
            sizes = {k: len(b) for k, (_, b) in parts.items()}
            print(f"{name}: {len(parts) // 3} shards from {f}, RMS error against BF16 {rms:.3e}", flush=True)
        else:   # written by an earlier run: its header, through the mount (the first MiB)
            head, _ = st_read(open(dst, "rb").read(1 << 20))
            sizes = {k: e["data_offsets"][1] - e["data_offsets"][0] for k, e in head.items() if k != "__metadata__"}
        for k, n in sizes.items():
            wmap[k] = name
            nbytes += n
    json.dump({"source": f"{SRC}@{SRC_REV}", "bytes": nbytes, "weight_map": wmap}, open(os.path.join(out_dir, "ple8.json"), "w"),
              indent=1)
    print(f"{out_dir}: {len(wmap) // 3} shards in {len(files)} files")


def pack(repo, out_dir):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
    api, man = HfApi(), json.load(open(os.path.join(out_dir, "ple8.json")))
    rev = api.model_info(repo).sha
    get = lambda f: hf_hub_download(repo, f, revision=rev, local_dir=os.path.join(WORK, "pk"))
    idx, cfg = json.load(open(get("model.safetensors.index.json"))), json.load(open(get("config.json")))
    old_files = sorted({f for k, f in idx["weight_map"].items() if OLD.search(k)})
    modules = sorted({k.rsplit(".", 1)[0] for k in man["weight_map"]})
    meta = api.get_safetensors_metadata(repo, revision=rev)   # the headers, by range requests
    total = sum(e.data_offsets[1] - e.data_offsets[0] for fm in meta.files_metadata.values() for e in fm.tensors.values())
    old = sum(e.data_offsets[1] - e.data_offsets[0] for f in old_files for k, e in meta.files_metadata[f].tensors.items() if OLD.search(k))
    ni = new_index(idx, man["weight_map"], total - old + man["bytes"])
    up = os.path.join(WORK, "up")
    os.makedirs(up, exist_ok=True)
    open(os.path.join(up, "config.json"), "w").write(json.dumps(new_config(cfg, modules), indent=2))
    open(os.path.join(up, "model.safetensors.index.json"), "w").write(json.dumps(ni, indent=2))
    pr = api.create_commit(repo, [CommitOperationAdd(f, os.path.join(up, f)) for f in ("config.json", "model.safetensors.index.json")],
                           commit_message="The 8-bit PLE table, group 32, from BF16 (plan ple-8bit): config and index",
                           create_pr=True, parent_commit=rev)
    ref = pr.pr_revision
    print(f"{repo}: {pr.pr_url} from {rev[:8]}", flush=True)
    for f in old_files:   # the shards that hold the 4-bit table: rewritten without it, or deleted
        p = get(f)
        rest = without(open(p, "rb").read(), lambda k: bool(OLD.search(k)))
        os.remove(p)
        if rest is None:
            op = CommitOperationDelete(f)
        else:
            open(os.path.join(up, f), "wb").write(rest)
            op = CommitOperationAdd(f, os.path.join(up, f))
        api.create_commit(repo, [op], commit_message=f"{f} without the 4-bit table", revision=ref)
        if rest is not None:
            os.remove(os.path.join(up, f))
        print(f"  {f}: {'deleted' if rest is None else 'rewritten'}", flush=True)
    names = sorted(set(man["weight_map"].values()))
    for i in range(0, len(names), 6):   # the 8-bit files, six a commit (Xet sends each chunk once across the packs)
        batch = names[i:i + 6]
        for f in batch:
            hf_cp(f"{URI}/{f}", os.path.join(up, f))
        api.create_commit(repo, [CommitOperationAdd(f, os.path.join(up, f)) for f in batch],
                          commit_message=f"The 8-bit PLE table, files {i + 1} to {i + len(batch)}", revision=ref)
        for f in batch:
            os.remove(os.path.join(up, f))
        print(f"  {batch[0]} .. {batch[-1]}", flush=True)
    print(f"{repo}: {pr.pr_url} ready (not merged)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "table":
        table(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "pack":
        pack(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
