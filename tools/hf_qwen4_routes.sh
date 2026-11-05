#!/usr/bin/env bash
# tools/hf_qwen4_routes.sh - routing traces (Q4E_ROUTES) of a Qwen3.8-Flash-Next pack on calibration texts, for
# tools/qwen4_tier_map.py (RAM-tier plan T4): chat (UltraChat test conversations in the chat template), code (this
# repository's C sources) and wiki (wikitext-103 train: other articles than the wikitext-2 test text of hf_llama_kld.sh).
#   SRC=/b/src OUT=/b/traces [TEXTS="chat code"] [PACK=REPO@REV] [CHARS=60000] bash tools/hf_qwen4_routes.sh
set -euo pipefail
PACK=${PACK:-txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3@5753a67}
CHARS=${CHARS:-60000}
TEXTS=${TEXTS:-"chat code"}
export DEBIAN_FRONTEND=noninteractive HF_XET_HIGH_PERFORMANCE=1
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq cmake libicu-dev python3-pip >/dev/null 2>&1
pip install -q --break-system-packages "huggingface_hub[hf_xet]" pyarrow >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; uptime
mkdir -p "$OUT"
hf download "${PACK%@*}" --revision "${PACK#*@}" --local-dir /model --max-workers 32 >/dev/null &
rm -rf /work && cp -r "$SRC" /work
cmake -S /work -B /work/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null
cmake --build /work/build -j "$(nproc)" --target nslm-qwen4-gen 2>&1 | grep -E "error" || true
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset --include "data/test_sft-*" --local-dir /uc >/dev/null
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-103-raw-v1/train-00000-*" --local-dir /wt >/dev/null
python3 - "$CHARS" <<'EOF'
import glob, sys, pyarrow.parquet as pq
n = int(sys.argv[1])
t = pq.read_table(glob.glob("/uc/data/test_sft-*")[0]).column("messages").to_pylist()
chat = "".join("".join("<|im_start|>%s\n%s<|im_end|>\n" % (m["role"], m["content"]) for m in c) for c in t)
open("/chat.txt", "w").write(chat[:n])
code = "".join(open(f).read() for f in sorted(glob.glob("/work/engine/*.c") + glob.glob("/work/nslm/*.c")))
open("/code.txt", "w").write(code[:n])
wiki = "".join(pq.read_table(glob.glob("/wt/wikitext-103-raw-v1/train-00000-*")[0]).column("text").to_pylist()[:20000])
open("/wiki.txt", "w").write(wiki[:n])
EOF
wait
for t in $TEXTS; do
    Q4E_ROUTES="$OUT/p3-$t.bin" /work/build/bin/nslm-qwen4-gen --model /model --prompt "$(cat /$t.txt)" --raw --max-tokens 1 \
        --ctx 32768 >"$OUT/p3-$t.log" 2>&1 || { echo "== $t failed"; tail -5 "$OUT/p3-$t.log"; continue; }
    echo "== $t: $(grep -m1 -i -E "prompt" "$OUT/p3-$t.log") $(du -h "$OUT/p3-$t.bin" | cut -f1)"
done
