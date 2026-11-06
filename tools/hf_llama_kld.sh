#!/usr/bin/env bash
# tools/hf_llama_kld.sh - KL divergence of Qwen3.8-Flash-Next packs against BF16 in llama.cpp, each with the BF16 PLE
# table (RAM-tier plan T1), on an HF Jobs node with 1 TB of RAM and 384 GB of VRAM (rtx-pro-6000x4).
#
#   SRC=/b/src OUT=/b/kld [PACKS="Q8_0 P3 P4 UD-IQ3_XXS UD-Q3_K_XL UD-IQ4_XS UD-Q4_K_XL"] [CHUNKS=64] [MAPS=/b/maps] \
#     [S3BANK=/b/s3/experts-s3] [TQBANK=/b/tq] [REF=/b/kld-ref] bash tools/hf_llama_kld.sh
#
# REF: a folder for the BF16 reference logits. The first job with REF writes REF/ref-cCHUNKS.kld. Later jobs copy it and
# skip BF16 (330 GB): then every pack keeps its own PLE table (Q8_0's adds 0.00005 of KL), and rtx-pro-6000x2 is enough.
#
# A pack is an Unsloth folder, a seed or trellis tier (P3, P4, S3, TQ{K}L{L}: the Q8_0 shards with those experts,
# nslm-qwen4-emulate; TQ3L12 is the folder TQBANK/TQ3L12 of nslm-moe --codec tq banks, as tools/hf_seedbank.sh writes), or
# LOW+HIGH@NAME: experts from two tiers by the tier map MAPS/NAME.bin (tools/qwen4_tier_map.py), e.g. S3+P3@down, or
# Q8_0-PLE: Q8_0 with its own Q8_0 PLE table in place of BF16's (the PLE format's own KL; it must come before the tiers).
# NAME/mlx: a tier with P3's 4-bit MLX dense tensors too (the pack as shipped, nslm-qwen4-emulate --dense), or /PATH: a
# packed folder (nslm-qwen4-pack --tq) with its own experts and dense tensors.  These come last: they change the Q8_0
# dense tensors in place.  The reference logits: BF16 on the first CHUNKS windows of 2,048
# tokens of wikitext-2 (test).
set -uo pipefail
UREPO=unsloth/Qwen3.8-Flash-Next-GGUF
UREV=${UREV:-d96201194f103c906994f283f504209dc8d2a10c}
LLAMA=${LLAMA:-5ce06fc71f5e938301f7dc51697a59ae308bd6b7}
PACKS=${PACKS:-"Q8_0 P3 P4 UD-IQ3_XXS UD-Q3_K_XL UD-IQ4_XS UD-Q4_K_XL"}
CHUNKS=${CHUNKS:-64}
declare -A SEED=([P3]=txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3@5753a67 [P4]=txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4@4157e09)
export DEBIAN_FRONTEND=noninteractive HF_XET_HIGH_PERFORMANCE=1
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq cmake git libicu-dev python3-pip >/dev/null 2>&1
pip install -q --break-system-packages "huggingface_hub[hf_xet]" pyarrow >/dev/null 2>&1
mkdir -p "$OUT" /m /w
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; uptime; free -g | head -2; df -h / | tail -1

seed() {   # seed P3|P4|S3|TQ*: that tier in /m/seed-TIER, kept for the next packs (S3, TQk: banks from the bucket)
    [ -d "/m/seed-$1" ] && return
    local t0; t0=$(date +%s)
    if [ "$1" = S3 ]; then mkdir -p /m/seed-S3 && cp "${S3BANK:-/b/s3/experts-s3}"/*.blks /m/seed-S3/
    elif [[ $1 == TQ* ]]; then mkdir -p "/m/seed-$1" && cp "${TQBANK:-/b/tq}/$1"/*.blkt "/m/seed-$1/"
    else hf download "${SEED[$1]%@*}" --revision "${SEED[$1]#*@}" --local-dir "/m/seed-$1" --max-workers 32 >/dev/null; fi
    echo "== $1: fetched in $(( $(date +%s) - t0 )) s, $(du -sh "/m/seed-$1" | cut -f1)"
}
get() {   # get FOLDER [EXCLUDED SHARD]: one Unsloth folder into /m
    local t0; t0=$(date +%s)
    hf download "$UREPO" --revision "$UREV" --include "$1/*" ${2:+--exclude "$1/*$2.gguf"} --local-dir /m --max-workers 32 >/dev/null
    echo "== $1: download $(( $(date +%s) - t0 )) s, $(du -sh "/m/$1" | cut -f1)"
}
stamp() {   # stamp FILE NO COUNT: the split.no / split.count of a shard (as Unsloth's engram/set_split.py)
    python3 -I - "$@" <<'EOF'
import struct, sys
path, no, cnt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "r+b") as f:
    head = f.read(1 << 24)
    for key, val in ((b"split.no", no - 1), (b"split.count", cnt)):
        f.seek(head.index(struct.pack("<Q", len(key)) + key + struct.pack("<I", 2)) + len(key) + 12)
        f.write(struct.pack("<H", val))
EOF
}
ppl() {   # ppl MODEL_DIR LOG ARGS...: llama-perplexity on the first shard of the folder
    local first; first=$(ls "$1"/*-00001-of-*.gguf)
    shift
    local log=$1; shift
    /w/llama/build/bin/llama-perplexity -m "$first" -c 2048 -b 2048 -ub 2048 --chunks "$CHUNKS" -ngl 999 "$@" >"$log" 2>&1
    echo "rc $? $(date -u +%T)"; grep -E "Mean    KLD|Maximum KLD|99.0%   KLD|Median  KLD|Same top p|Mean PPL|Final estimate|error|failed" "$log" | head -12
}

t0=$(date +%s)   # llama.cpp (CUDA, this node's GPUs) and nslm-qwen4-emulate
git clone -q https://github.com/ggml-org/llama.cpp /w/llama && git -C /w/llama checkout -q "$LLAMA"
cmake -S /w/llama -B /w/llama/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release >/w/llama-cmake.log 2>&1
cmake --build /w/llama/build -j "$(nproc)" --target llama-perplexity >/w/llama-build.log 2>&1 || { tail -30 /w/llama-build.log; exit 1; }
rm -rf /w/src && cp -r "$SRC" /w/src
cmake -S /w/src -B /w/src/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null 2>&1
cmake --build /w/src/build -j "$(nproc)" --target nslm-qwen4-emulate 2>&1 | grep -E "error" || true
echo "== builds: $(( $(date +%s) - t0 )) s"
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/test-*" --local-dir /w/wt >/dev/null
python3 -c 'import glob, pyarrow.parquet as pq; open("/w/text.txt", "w").write("".join(pq.read_table(glob.glob("/w/wt/wikitext-2-raw-v1/test-*")[0]).column("text").to_pylist()))'

SAVED=${REF:+$REF/ref-c$CHUNKS.kld}
if [ -n "$SAVED" ] && [ -f "$SAVED" ]; then   # the saved reference: no BF16, each pack's own PLE table
    cp "$SAVED" /w/ref.kld && echo "== BF16: the saved reference $SAVED"; PLE=
else
    get BF16   # the reference, with its own BF16 table (shard 3 of 8), which every pack then uses
    ppl /m/BF16 "$OUT/BF16.log" -f /w/text.txt --kl-divergence-base /w/ref.kld
    [ -n "$SAVED" ] && mkdir -p "$REF" && cp /w/ref.kld "$SAVED" && echo "== BF16: reference saved to $SAVED"
    PLE=/m/BF16/Qwen3.8-Flash-Next-BF16-00003-of-00008.gguf
fi
for pk in $PACKS; do
    case $pk in
        Q8_0) if [ -z "$PLE" ]; then [ -d /m/Q8_0 ] || get Q8_0
              else [ -d /m/Q8_0 ] || get Q8_0 -00003-of-00006; ln -sf "$PLE" /m/Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf; stamp "$PLE" 3 6; fi
            dir=/m/Q8_0 ;;
        Q8_0-PLE)   # Q8_0's shards with its own table: shard 3 from Unsloth, the others linked from /m/Q8_0
            mkdir -p /m/Q8_0-PLE && hf download "$UREPO" --revision "$UREV" --include "Q8_0/*-00003-of-00006.gguf" --local-dir /m/q8ple >/dev/null
            for f in /m/Q8_0/*.gguf; do ln -sf "$f" /m/Q8_0-PLE/; done
            ln -sf /m/q8ple/Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf /m/Q8_0-PLE/; dir=/m/Q8_0-PLE ;;
        P3|P4|S3|TQ*|*+*@*|*/mlx|/*)   # the experts of the Q8_0 folder replaced: Q8_0 must come first
            t1=$(date +%s) base=${pk%/mlx}
            if [[ -n ${MLX:-} && $pk != */mlx && $pk != /* ]]; then echo "== $pk: after a /mlx or folder pack the dense tensors changed: put those last"; continue; fi
            if [[ $pk == /* ]]; then args=(--pack "$pk" --dense "$pk"); MLX=1
            elif [[ $base == *@* ]]; then
                lo=${base%%+*} hi=${base#*+}; hi=${hi%@*}
                seed "$lo"; seed "$hi"; args=(--pack "/m/seed-$lo" --pack "/m/seed-$hi" --map "$MAPS/${base#*@}.bin")
            else seed "$base"; args=(--pack "/m/seed-$base"); fi
            if [[ $pk == */mlx ]]; then seed P3; args+=(--dense /m/seed-P3); MLX=1; fi
            if ! /w/src/build/bin/nslm-qwen4-emulate "${args[@]}" --workers "$(nproc)" /m/Q8_0/*.gguf | grep -E "experts:|of the pack|dense"; then
                echo "== $pk: emulate failed"; continue
            fi
            [ -n "$PLE" ] && stamp "$PLE" 3 6
            echo "== $pk: emulate $(( $(date +%s) - t1 )) s"; dir=/m/Q8_0 ;;
        UD-*) if [ -z "$PLE" ]; then get "$pk"
              else get "$pk" -00004-of-00004; stamp "$PLE" 4 4; ln -sf "$PLE" "/m/$pk/Qwen3.8-Flash-Next-$pk-00004-of-00004.gguf"; fi
            dir=/m/$pk ;;
        *) echo "unknown pack $pk"; continue ;;
    esac
    echo "== $pk"
    ppl "$dir" "$OUT/${pk//[+@\/]/_}.log" --kl-divergence-base /w/ref.kld --kl-divergence
    [[ $pk == UD-* ]] && rm -rf "/m/$pk"
done
uptime
