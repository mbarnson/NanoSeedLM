#!/usr/bin/env bash
# tools/hf_qwen4_gen.sh - a Qwen3.8 pack through the CUDA engine on an HF Jobs GPU node: copies the pack from a mounted
# bucket to local disk, builds nslm-qwen4-gen and generates.  SRC: the tree; PACK: the pack folder; GEN_ARGS: more options.
#
#   SRC=/b/src PACK=/b/pack-p4 PROMPT="Why do tide pools matter?" [GEN_ARGS="--vram-experts"] bash tools/hf_qwen4_gen.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v cmake >/dev/null || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq cmake libicu-dev >/dev/null 2>&1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2
t0=$(date +%s)
mkdir -p /model && cp -r "$PACK"/. /model/
echo "pack copy: $(( $(date +%s) - t0 )) s, $(du -sh /model | cut -f1)"
rm -rf /work && cp -r "$SRC" /work
cmake -S /work -B /work/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null
cmake --build /work/build -j "$(nproc)" --target nslm-qwen4-gen 2>&1 | grep -E "error|warning: " || true
/work/build/bin/nslm-qwen4-gen --model /model --prompt "$PROMPT" --max-tokens "${MAX_TOKENS:-96}" ${GEN_ARGS:-}
