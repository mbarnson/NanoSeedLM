#!/usr/bin/env bash
# tools/hf_release_publish.sh - release step R5: each folder of tools/hf_release_pack.sh (OUT/NAME), then its card
# (CARDS/NAME.md as README.md), to txgsync/Qwen3.8-Flash-Next-NanoSeedLM-NAME, private.  Run by the account that owns the
# repositories, with its token as a job secret (the data goes from the bucket to the Hub, not over the caller's link):
#   hf jobs run --flavor cpu-xl --timeout 4h --secrets HF_TOKEN -v hf://buckets/txgsync/nslm-qwen38:/b \
#       [-e PACKS="TQ3 TQ3-D4 TQ4"] python:3.12 bash /b/src-REV/tools/hf_release_publish.sh
# Make a repository public later on its settings page, after the download tests.
set -euo pipefail
OUT=${OUT:-/b/release} CARDS=${CARDS:-/b/release/cards} PACKS=${PACKS:-"TQ3 TQ3-D4 TQ4"}
pip install -q "huggingface_hub[hf_xet]" >/dev/null 2>&1
for P in $PACKS; do
    R=txgsync/Qwen3.8-Flash-Next-NanoSeedLM-$P
    [ -f "$CARDS/$P.md" ] && [ -f "$OUT/$P/config.json" ] || { echo "$P: no card $CARDS/$P.md or no folder $OUT/$P"; exit 1; }
    hf repos create "$R" --repo-type model --private --exist-ok >/dev/null
    t0=$(date +%s)   # the folder from the mounted bucket, then the card (resumes when re-run)
    hf upload "$R" "$OUT/$P" --repo-type model --exclude "*.log" --commit-message "Qwen3.8-Flash-Next-NanoSeedLM-$P (release 1)"
    hf upload "$R" "$CARDS/$P.md" README.md --repo-type model --commit-message "Qwen3.8-Flash-Next-NanoSeedLM-$P: card"
    echo "$R: uploaded in $(( $(date +%s) - t0 )) s, $(python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('$R').sha)")"
done
