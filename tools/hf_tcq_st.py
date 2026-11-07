#!/usr/bin/env python3
"""tools/hf_tcq_st.py - plan tcq-safetensors TS5: a trellis pack's .blkt banks as model-tcq-*.safetensors shards
(nslm-qwen4-pack --tcq-st), with the MLX loader files of this commit and the card's new file list, as a pull request on
the repository (not merged). The history stays, so the old revisions (the v0.1.1 start scripts' pins) still download.

  python3 tools/hf_tcq_st.py REPO REVISION PACK      PACK: the nslm-qwen4-pack binary of this commit

As an HF job, by the account that owns the packs (its token a job secret). cpu-xl has 1,000 GB of disk, and a pack and
its new shards need 175 GB, one pack at a time:
  hf jobs run --flavor cpu-xl --timeout 4h --secrets HF_TOKEN -e SRC=COMMIT nvidia/cuda:12.8.1-devel-ubuntu22.04 bash -c '
      apt-get update -qq && apt-get install -y -qq git cmake ninja-build libicu-dev python3-pip >/dev/null &&
      pip3 install -q "huggingface_hub[hf_xet]" && git clone -q https://github.com/mbarnson/NanoSeedLM /src && cd /src &&
      git checkout -q $SRC && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null &&
      cmake --build build --target nslm-qwen4-pack >/dev/null && for x in TCQ3=REV TCQ3-D4=REV TCQ4=REV; do
      python3 tools/hf_tcq_st.py txgsync/Qwen3.8-Flash-Next-NanoSeedLM-${x%%=*} ${x#*=} build/bin/nslm-qwen4-pack; done'
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOADER = ("nanoseedlm_qwen4.py", "nanoseedlm_seeds.py", "nanoseedlm_trellis.py")
WORK = Path(os.environ.get("WORK", "/w"))
# The card's file list: the banks' bullet, and the loader's words for them (tools/qwen4_release_card.py says the same)
OLD_BANKS = re.compile(r"- `L\{layer\}_\{gate_proj,up_proj,down_proj\}\.blkt`: the trellis-coded routed experts of layers 0 to 47 and of the MTP\n"
                       r"  head \(layer 48\)\. Read \[the format\]\(([^)]*)\)\.\n")
NEW_BANKS = ("- `model-tcq-*.safetensors`: the trellis-coded routed experts of layers 0 to 47 and of the MTP head, one shard\n"
             "  a layer. Each projection has the entries `NAME.tcq_words`, `NAME.tcq_scales`, `NAME.tcq_params` and\n"
             "  `NAME.tcq_err`. Read [the format](\\1).\n")
OLD_LOADER, NEW_LOADER = "puts the trellis banks in place of the routed experts.", "puts the trellis experts in place of the routed experts."


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(repo: str, rev: str, pack: str) -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, snapshot_download
    api = HfApi()
    d = WORK / repo.split("/")[1]
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.time()
    snapshot_download(repo, revision=rev, local_dir=d, max_workers=16)
    shutil.rmtree(d / ".cache", ignore_errors=True)
    banks = sorted(p.name for p in d.glob("L*_*_proj.blkt"))
    print(f"{repo}@{rev[:12]}: {len(banks)} banks, downloaded in {time.time() - t0:.0f} s", flush=True)
    assert len(banks) == 147, f"{repo}: {len(banks)} .blkt banks, not 147"
    t0 = time.time()
    subprocess.run([pack, "--tcq-st", str(d), "--loader", str(ROOT / "tools" / LOADER[0])], check=True)
    shards = sorted(p.name for p in d.glob("model-tcq-*.safetensors"))
    assert len(shards) == 49 and not list(d.glob("*.blkt")), f"{repo}: {len(shards)} shards after the conversion"
    print(f"{repo}: converted in {time.time() - t0:.0f} s", flush=True)
    card = (d / "README.md").read_text()
    new_card, n = OLD_BANKS.subn(NEW_BANKS, card)
    assert n == 1 and OLD_LOADER in new_card, f"{repo}: the card's file list is not the expected one"
    (d / "README.md").write_text(new_card.replace(OLD_LOADER, NEW_LOADER))
    with ThreadPoolExecutor(8) as ex:   # the new files' SHA-256, for the gate TB against the M4 Max conversion
        for name, h in zip(shards, ex.map(lambda s: sha256(d / s), shards)):
            print(f"{h}  {name}")
    print(f"{sha256(d / 'model.safetensors.index.json')}  model.safetensors.index.json", flush=True)
    ops = [CommitOperationAdd(f, str(d / f)) for f in shards + ["model.safetensors.index.json", "README.md", *LOADER]]
    ops += [CommitOperationDelete(b) for b in banks]
    t0 = time.time()
    c = api.create_commit(repo, ops, parent_commit=rev, create_pr=True,
                          commit_message="Trellis experts in model-tcq-*.safetensors shards, not .blkt banks (plan tcq-safetensors)",
                          commit_description=f"nslm-qwen4-pack --tcq-st, NanoSeedLM {os.environ.get('SRC', '?')}. The bytes of the "
                                             "codes and the scales do not change. The other shards do not change.")
    print(f"{repo}: pull request {c.pr_url} ({c.oid}), uploaded in {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(d)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
