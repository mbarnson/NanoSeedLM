#!/usr/bin/env bash
# tools/hf_seedbank.sh - seed searches on a multi-GPU Linux node (HF Jobs): builds the tree with CMake, then runs one
# process per GPU through every search in JOBS, each GPU on its --shard of the tensors.  Finished files are skipped, so
# a rerun resumes.  Logs and the search outputs go to OUT.
#
#   SRC=/src MODEL=/model ACT=/in/actsq_moe.bin XTX=/in/xtx OUT=/out [JOBS="moe3 dense3 dense4"] [GPUS=n] \
#     [WORKERS=8] [MOE_EXTRA="--layers 20-20"] [DENSE_EXTRA="--only layers.20."] bash tools/hf_seedbank.sh
#
# JOBS: moe3 / moe4 routed and value experts (nslm-moe --scope all, AW) at P = 3 / 4; dense3 / dense4 / dense8 the dense
# tensors (nslm-dense --mode gptq) at P = 3 / 4 / 8.  MOE_EXTRA / DENSE_EXTRA: more options (e.g. one layer, for a benchmark).
set -euo pipefail
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
JOBS=${JOBS:-"moe3 dense3 dense4"}
WORKERS=${WORKERS:-8}
MOE_EXTRA=${MOE_EXTRA:-}
DENSE_EXTRA=${DENSE_EXTRA:-}
mkdir -p "$OUT/log"
if ! command -v cmake >/dev/null || [ ! -e /usr/include/unicode/unorm2.h ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq cmake libicu-dev >/dev/null 2>&1
fi
rm -rf /work && cp -r "$SRC" /work
cmake -S /work -B /work/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null
cmake --build /work/build -j "$(nproc)" --target nslm-moe nslm-dense 2>&1 | grep -E "error" || true
BIN=/work/build/bin
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee "$OUT/log/gpus.csv"

run_gpu() {   # one GPU: its shard of every job, in order
    local g=$1 t0 j
    for j in $JOBS; do
        t0=$(date +%s)
        case $j in
            moe3) "$BIN/nslm-moe" --model "$MODEL" --act "$ACT" --out "$OUT/experts-p3" --scope all --shard "$g/$GPUS" --workers "$WORKERS" $MOE_EXTRA ;;
            moe4) "$BIN/nslm-moe" --model "$MODEL" --act "$ACT" --out "$OUT/experts-p4" --scope all --p4 --shard "$g/$GPUS" --workers "$WORKERS" $MOE_EXTRA ;;
            dense3|dense4|dense8)
                "$BIN/nslm-dense" --model "$MODEL" --xtx "$XTX" --out "$OUT/dense-p${j#dense}" --mode gptq --codec "p${j#dense}" \
                    --shard "$g/$GPUS" --workers "$WORKERS" $DENSE_EXTRA ;;
            *) echo "unknown job $j"; return 1 ;;
        esac
        echo "== $j gpu $g: $(( $(date +%s) - t0 )) s"
    done
}
pids=()
for g in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$g run_gpu "$g" >"$OUT/log/gpu$g.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h "^== " "$OUT"/log/gpu*.log
exit $rc
