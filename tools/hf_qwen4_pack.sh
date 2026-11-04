#!/usr/bin/env bash
# tools/hf_qwen4_pack.sh - nslm-qwen4-pack on an HF Jobs CPU node (image python:3.12): downloads the MLX base folder,
# packs it with the seed blocks in BLK (a mounted bucket), then copies the folder to OUT (a mounted bucket).
#
#   SRC=/src BASE=REPO@REVISION BLK=/in/p4/experts-p4 P=4 OUT=/out/pack-p4 bash tools/hf_qwen4_pack.sh
set -euo pipefail
pip install -q "huggingface_hub[hf_xet]" >/dev/null 2>&1
t0=$(date +%s)
hf download "${BASE%@*}" --revision "${BASE#*@}" --local-dir /base --max-workers 16 >/dev/null
echo "base download: $(( $(date +%s) - t0 )) s, $(du -sh /base | cut -f1)"
cc -std=c99 -O2 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -I"$SRC/nslm" "$SRC/nslm/qwen4_pack.c" \
    "$SRC/nslm/lib_format.c" "$SRC/nslm/lib_json.c" "$SRC/nslm/lib_model_st.c" "$SRC/nslm/lib_moe.c" -o /tmp/nslm-qwen4-pack -lm
t0=$(date +%s)
/tmp/nslm-qwen4-pack --base /base "$([ "$P" = 4 ] && echo --blk4 || echo --blk)" "$BLK" --out /pack --loader "$SRC/tools/nanoseedlm_qwen4.py"
echo "pack: $(( $(date +%s) - t0 )) s, $(du -sh /pack | cut -f1)"
t0=$(date +%s)
mkdir -p "$OUT" && cp -r /pack/. "$OUT"/
echo "copy to $OUT: $(( $(date +%s) - t0 )) s"
