#!/usr/bin/env bash
# tools/hf_release_publish.sh - release step R5: each folder of tools/hf_release_pack.sh (OUT/NAME), then its card
# (CARDS/NAME.md as README.md), to txgsync/Qwen3.8-Flash-Next-NanoSeedLM-NAME, private.  Run by the account that owns the
# repositories, with its token as a job secret (the data goes from the bucket to the Hub, not over the caller's link).
# Each folder goes to the local disk first: hf upload of the mounted bucket fails with EIO, as a memory map gives SIGBUS:
#   hf jobs run --flavor cpu-xl --timeout 4h --secrets HF_TOKEN -v hf://buckets/txgsync/nslm-qwen38:/b \
#       [-e PACKS="TCQ3 TCQ3-D4 TCQ4"] python:3.12 bash /b/src-REV/tools/hf_release_publish.sh
# Make a repository public later on its settings page, after the download tests.
set -euo pipefail
OUT=${OUT:-/b/release} CARDS=${CARDS:-/b/release/cards} PACKS=${PACKS:-"TCQ3 TCQ3-D4 TCQ4"}
pip install -q "huggingface_hub[hf_xet]" >/dev/null 2>&1
for P in $PACKS; do
    R=txgsync/Qwen3.8-Flash-Next-NanoSeedLM-$P
    [ -f "$CARDS/$P.md" ] && [ -f "$OUT/$P/config.json" ] || { echo "$P: no card $CARDS/$P.md or no folder $OUT/$P"; exit 1; }
    for f in nanoseedlm_qwen4.py nanoseedlm_seeds.py nanoseedlm_trellis.py LICENSE; do   # mlx-vlm and oMLX load these
        [ -f "$OUT/$P/$f" ] || { echo "$P: no $f (nslm-qwen4-pack --tq --loader writes the MLX files)"; exit 1; }
    done
    grep -q '"model_file"' "$OUT/$P/config.json" || { echo "$P: config.json has no model_file"; exit 1; }
    hf repos create "$R" --repo-type model --private --exist-ok >/dev/null
    t0=$(date +%s)   # cpu-xl has 1,000 GB of disk, the largest folder is 95 GiB
    rm -rf /pub && mkdir -p /pub && cp -r "$OUT/$P" /pub/ && cp "$CARDS/$P.md" "/pub/$P/README.md" && rm -f /pub/"$P"/*.log
    echo "$P: copied in $(( $(date +%s) - t0 )) s, $(du -sh "/pub/$P" | cut -f1), $(ls "/pub/$P" | wc -l) files"
    t0=$(date +%s)   # one folder, the card in it (resumes when re-run)
    hf upload "$R" "/pub/$P" --repo-type model --commit-message "Qwen3.8-Flash-Next-NanoSeedLM-$P (release 1)"
    echo "$R: uploaded in $(( $(date +%s) - t0 )) s, $(python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('$R').sha)")"
done
rm -rf /pub
