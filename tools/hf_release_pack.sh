#!/usr/bin/env bash
# tools/hf_release_pack.sh - the trellis release folders (docs/plans/2026-11-05-release-family.md, steps R4 and R3) on an HF
# Jobs GPU node: P3's folder with oQ8e-mtp's 8-bit dense modules and each pack's banks (nslm-qwen4-pack --tq --dense8) into
# OUT/NAME, then for each a local copy, two runs of nslm-qwen4-gen and the KL as shipped (tools/hf_llama_kld.sh /PATH
# packs).  PACK=0 tests the folders that an earlier run packed; TESTS (default PACKS) names the folders to test.
#   hf jobs run --flavor rtx-pro-6000x2 -v hf://buckets/txgsync/nslm-qwen38:/b -e SRC=/b/src-REV [-e OUT=/b/release] \
#       [-e PACKS="TCQ3 TCQ3-D4 TCQ4"] nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /b/src-REV/tools/hf_release_pack.sh
set -euo pipefail
OUT=${OUT:-/b/release} TQBANK=${TQBANK:-/b/tq} PACKS=${PACKS:-"TCQ3 TCQ3-D4 TCQ4"} TESTS=${TESTS:-${PACKS:-"TCQ3 TCQ3-D4 TCQ4"}}
P3=txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3 P3REV=5753a67
OQ8=mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp OQ8REV=2a9de025436ea977720e4e3d6f369b185838e791
PROMPT="Write a Python function that returns the n-th Fibonacci number, with a docstring."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq cmake ninja-build git libicu-dev pkg-config clang python3-pip >/dev/null
pip install -q --break-system-packages "huggingface_hub[hf_xet]" >/dev/null 2>&1
t0=$(date +%s)
rm -rf /w/src && mkdir -p /w /m && cp -r "$SRC" /w/src
cmake -S /w/src -B /w/src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null
cmake --build /w/src/build --target nslm-qwen4-pack nslm-qwen4-gen 2>&1 | grep -E "error" || true
mkdir -p /w/bin && cp /w/src/build/bin/nslm-qwen4-pack /w/src/build/bin/nslm-qwen4-gen /w/bin/   # hf_llama_kld.sh rebuilds /w/src
echo "== builds: $(( $(date +%s) - t0 )) s"

if [ "${PACK:-1}" = 1 ]; then   # PACK=0: test folders that an earlier run packed into OUT
t0=$(date +%s)   # P3 whole (/m/seed-P3, as hf_llama_kld.sh names it), oQ8e only its shards with dense modules
hf download "$P3" --revision "$P3REV" --local-dir /m/seed-P3 --max-workers 32 >/dev/null
hf download "$OQ8" config.json model.safetensors.index.json --revision "$OQ8REV" --local-dir /m/oq8 >/dev/null
mapfile -t SHARDS < <(python3 -I -c '
import json, sys
m = json.load(open("/m/oq8/model.safetensors.index.json"))["weight_map"]
print("\n".join(sorted({f for k, f in m.items() if ".switch_mlp." not in k and ".ple." not in k})))')
hf download "$OQ8" "${SHARDS[@]}" --revision "$OQ8REV" --local-dir /m/oq8 --max-workers 32 >/dev/null   # names, not --include
echo "== downloads: $(( $(date +%s) - t0 )) s, P3 $(du -sh /m/seed-P3 | cut -f1), oQ8e ${#SHARDS[@]} shards $(du -sh /m/oq8 | cut -f1)"
mkdir -p "$OUT"   # nslm-qwen4-pack makes the last level only
for p in $PACKS; do
    case $p in
        TCQ3) tq=TQ3L12 down= ;;
        TCQ3-D4) tq=TQ3L12 down=TQ4L12 ;;
        TCQ4) tq=TQ4L12 down= ;;
        *) echo "== $p: unknown pack"; continue ;;
    esac
    t0=$(date +%s)
    /w/bin/nslm-qwen4-pack --base /m/seed-P3 --tq "$TQBANK/$tq" ${down:+--tq-down "$TQBANK/$down"} --dense8 /m/oq8 --loader /w/src/tools/nanoseedlm_qwen4.py --out "$OUT/$p"
    echo "== $p: packed in $(( $(date +%s) - t0 )) s, $(du -sh "$OUT/$p" | cut -f1), $(ls "$OUT/$p" | wc -l) files"
done
rm -rf /m/oq8 /m/seed-P3
fi

first=1
for p in $TESTS; do   # each folder copied to the local disk: a memory map of the bucket mount faults (SIGBUS)
    t0=$(date +%s)
    rm -rf /m/rel && mkdir -p /m/rel && cp -r "$OUT/$p" /m/rel/
    echo "== $p: copied in $(( $(date +%s) - t0 )) s, $(du -sh "/m/rel/$p" | cut -f1)"
    for r in 1 2; do   # gate RC: the same text in two runs
        /w/bin/nslm-qwen4-gen --model "/m/rel/$p" --expert-cache-gb -1 --no-think --max-tokens 128 --prompt "$PROMPT" >"/w/$p-$r.txt" 2>"/w/$p-$r.err" || true
        echo "== $p run $r: md5 $(md5sum <"/w/$p-$r.txt" | cut -c1-8), $(grep -iE 'tok/s|tokens per second' "/w/$p-$r.err" | tail -1)"
    done
    cmp -s "/w/$p-1.txt" "/w/$p-2.txt" && [ -s "/w/$p-1.txt" ] && echo "== $p: the same text in both runs" || { echo "== $p: THE RUNS DIFFER OR FAILED"; tail -5 "/w/$p-1.err"; }
    head -c 600 "/w/$p-1.txt"; echo
    PACKS="$([ $first = 1 ] && echo Q8_0) /m/rel/$p" OUT="$OUT/kld" bash /w/src/tools/hf_llama_kld.sh   # Q8_0 once: it sets up /m/Q8_0
    first=0
done
