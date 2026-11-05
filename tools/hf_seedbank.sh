#!/usr/bin/env bash
# tools/hf_seedbank.sh - seed searches on a multi-GPU Linux node (HF Jobs): builds the tree with CMake, then runs one
# process per GPU through every search in JOBS, each GPU on its --shard of the tensors.  Finished files are skipped, so
# a rerun resumes.  Logs and the search outputs go to OUT.
#
# MODEL is a local folder, or REPO@REVISION (downloaded to local disk first: the Hub mount reads at a few MB/s, which
# bounds the expert searches).
#
#   SRC=/src MODEL=IFM/K2-Horizon-MoVA-36B-A4B@cca48b6 ACT=/in/actsq_moe.bin XTX=/in/xtx OUT=/out [JOBS="moe3 dense3 dense4"] [GPUS=n] \
#     [WORKERS=8] [LOG=$OUT/log] [MOE_EXTRA="--layers 20-20"] [DENSE_EXTRA="--only layers.20."] bash tools/hf_seedbank.sh
#
# Qwen3.8-Flash-Next: IMATRIX=FILE or REPO:FILE (a llama.cpp imatrix) in place of ACT, so moe3 / moe4 search gate, up
# and down (--scope gud).  ONLY=SUBSTR downloads only the shards that hold tensors named with SUBSTR (mlp.experts.).
#
# JOBS: moe3 / moe4 routed and value experts (nslm-moe --scope all, AW) at P = 3 / 4, moes3 at S3 (C = 12, P = 4, 3 bits a
# weight, .blks banks for nslm-qwen4-emulate); dense3 / dense4 / dense8 the dense
# tensors (nslm-dense --mode gptq) at P = 3 / 4 / 8; mla3 / mla4 / mla8 an MLA model's projections (nslm-moe --scope mla,
# GPTQ) at P = 3 / 4 / 8, after one capture on GPU 0 (nslm-mova-mlacapture over CALIB, the model being an MLA folder; the
# capture is kept in OUT/mla-capture and reused).  MOE_EXTRA / DENSE_EXTRA: more options (e.g. one layer, for a benchmark).
set -euo pipefail
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
JOBS=${JOBS:-"moe3 dense3 dense4"}
WORKERS=${WORKERS:-8}
MOE_EXTRA=${MOE_EXTRA:-}
DENSE_EXTRA=${DENSE_EXTRA:-}
LOG=${LOG:-$OUT/log}   # per-GPU logs (one directory per job when jobs share OUT); written locally, copied at the end
FINAL_LOG=$LOG
LOG=/tmp/seedbank-log
mkdir -p "$LOG" "$FINAL_LOG"
if ! command -v cmake >/dev/null || [ ! -e /usr/include/unicode/unorm2.h ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq cmake libicu-dev python3-pip >/dev/null 2>&1
fi
if [ ! -d "$MODEL" ]; then
    command -v pip >/dev/null || { apt-get install -y -qq python3-pip >/dev/null 2>&1; }
    pip install -q --break-system-packages "huggingface_hub[hf_xet]" >/dev/null 2>&1
    t0=$(date +%s)
    INCLUDE=()
    if [ -n "${ONLY:-}" ]; then   # the configs, and only the shards that hold matching tensors
        hf download "${MODEL%@*}" --revision "${MODEL#*@}" --include "*.json" --local-dir /model >/dev/null
        mapfile -t INCLUDE < <(python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["weight_map"]; print("*.json"); [print(f) for f in sorted({f for k,f in m.items() if sys.argv[2] in k})]' \
            /model/model.safetensors.index.json "$ONLY")
        INC=("${INCLUDE[@]}"); INCLUDE=()   # one --include for each pattern: more words after one would be positional files
        for f in "${INC[@]}"; do INCLUDE+=(--include "$f"); done
    fi
    hf download "${MODEL%@*}" --revision "${MODEL#*@}" "${INCLUDE[@]}" --local-dir /model --max-workers 16 >/dev/null
    echo "model download: $(( $(date +%s) - t0 )) s, $(du -sh /model | cut -f1)"
    MODEL=/model
fi
if [ -n "${IMATRIX:-}" ] && [ ! -f "$IMATRIX" ]; then   # REPO:FILE
    hf download "${IMATRIX%%:*}" "${IMATRIX#*:}" --local-dir /tmp/imatrix >/dev/null && IMATRIX="/tmp/imatrix/${IMATRIX#*:}"
fi
if [ -f "${ACT:-}" ]; then cp "$ACT" /tmp/act.bin && ACT=/tmp/act.bin; fi   # local copies: concurrent reads through a bucket
if [ -n "${IMATRIX:-}" ]; then CALIB_ARGS="--imatrix $IMATRIX --scope gud"; else CALIB_ARGS="--act $ACT --scope all"; fi
if [ -d "${XTX:-}" ]; then mkdir -p /tmp/xtx && cp -r "$XTX"/. /tmp/xtx/ && XTX=/tmp/xtx; fi   # mount can come back short
rm -rf /work && cp -r "$SRC" /work
cmake -S /work -B /work/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null
TARGETS=""   # only what JOBS needs (the MLA capture builds the whole CUDA engine)
for j in $JOBS; do case $j in moe*) TARGETS+=" nslm-moe" ;; dense*) TARGETS+=" nslm-dense" ;; mla*) TARGETS+=" nslm-moe nslm-mova-mlacapture" ;; esac; done
cmake --build /work/build -j "$(nproc)" --target $TARGETS 2>&1 | grep -E "error" || true
BIN=/work/build/bin
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee "$LOG/gpus.csv"

if [[ " $JOBS " == *" mla"* ]] && [ ! -f "$OUT/mla-capture/done" ]; then   # the MLA capture, once (GPU 0)
    mkdir -p "$OUT/mla-capture/xtx" /tmp/mlacap/xtx
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 "$BIN/nslm-mova-mlacapture" --model "$MODEL" --text "$CALIB" --out /tmp/mlacap/actsq_mla.bin --xtx /tmp/mlacap/xtx \
        >"$LOG/mlacapture.log" 2>&1
    cp -r /tmp/mlacap/. "$OUT/mla-capture/" && touch "$OUT/mla-capture/done"
    echo "== mla capture: $(( $(date +%s) - t0 )) s"
fi
[ -f "$OUT/mla-capture/done" ] && [ ! -d /tmp/mlacap ] && mkdir -p /tmp/mlacap && cp -r "$OUT/mla-capture/." /tmp/mlacap/

run_gpu() {   # one GPU: its shard of every job, in order
    local g=$1 t0 j
    for j in $JOBS; do
        t0=$(date +%s)
        case $j in
            moe3) "$BIN/nslm-moe" --model "$MODEL" $CALIB_ARGS --out "$OUT/experts-p3" --shard "$g/$GPUS" --workers "$WORKERS" $MOE_EXTRA ;;
            moe4) "$BIN/nslm-moe" --model "$MODEL" $CALIB_ARGS --out "$OUT/experts-p4" --p4 --shard "$g/$GPUS" --workers "$WORKERS" $MOE_EXTRA ;;
            moes3) "$BIN/nslm-moe" --model "$MODEL" $CALIB_ARGS --out "$OUT/experts-s3" --codec s3 --shard "$g/$GPUS" --workers "$WORKERS" $MOE_EXTRA ;;
            dense3|dense4|dense8)
                "$BIN/nslm-dense" --model "$MODEL" --xtx "$XTX" --out "$OUT/dense-p${j#dense}" --mode gptq --codec "p${j#dense}" \
                    --shard "$g/$GPUS" --workers "$WORKERS" $DENSE_EXTRA ;;
            mla3|mla4|mla8)
                "$BIN/nslm-moe" --model "$MODEL" --act /tmp/mlacap/actsq_mla.bin --scope mla --xtx /tmp/mlacap/xtx \
                    $([ "$j" = mla4 ] && echo --p4 || echo --codec "p${j#mla}") --out "$OUT/mla-p${j#mla}" --shard "$g/$GPUS" --workers "$WORKERS" ;;
            *) echo "unknown job $j"; return 1 ;;
        esac
        echo "== $j gpu $g: $(( $(date +%s) - t0 )) s"
    done
}
pids=()
for g in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$g run_gpu "$g" >"$LOG/gpu$g.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h "^== " "$LOG"/gpu*.log
tail -n 3 "$LOG"/gpu*.log
cp -r "$LOG"/. "$FINAL_LOG"/
exit $rc
