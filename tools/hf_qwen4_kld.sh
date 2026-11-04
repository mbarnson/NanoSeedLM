#!/usr/bin/env bash
# tools/hf_qwen4_kld.sh - nslm-qwen4-kld of the packs in PACKS (mounted folders) against the REF model (REPO@REVISION) on
# the wikitext-2 test text, on an HF Jobs GPU node.  SRC=/b/src REF=REPO@REV PACKS="/b/pack-p4 /b/pack-p3" bash tools/hf_qwen4_kld.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v cmake >/dev/null || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq cmake libicu-dev python3-pip >/dev/null 2>&1; }
pip install -q --break-system-packages "huggingface_hub[hf_xet]" pyarrow >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; uptime
t0=$(date +%s)
hf download "${REF%@*}" --revision "${REF#*@}" --local-dir /ref --max-workers 16 >/dev/null
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/test-*" --local-dir /wt >/dev/null
python3 -c 'import glob, pyarrow.parquet as pq; open("/text.txt", "w").write("".join(pq.read_table(glob.glob("/wt/wikitext-2-raw-v1/test-*")[0]).column("text").to_pylist()))'
echo "downloads: $(( $(date +%s) - t0 )) s"
rm -rf /work && cp -r "$SRC" /work
cmake -S /work -B /work/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null
cmake --build /work/build -j "$(nproc)" --target nslm-qwen4-kld 2>&1 | grep -E "error" || true
for p in $PACKS; do
    rm -rf /model && mkdir -p /model && cp -r "$p"/. /model/
    echo "== $p"
    /work/build/bin/nslm-qwen4-kld --ref /ref --model /model --text /text.txt --tokens "${TOKENS:-1024}" ${KLD_ARGS:-}
done
