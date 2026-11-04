#!/usr/bin/env bash
# tools/hf_cuda_test.sh - builds and runs CUDA test targets on an HF Jobs GPU node (image nvidia/cuda:*-devel).
#   SRC=/src [TARGETS="test_qwen4_kernels test_qwen4_engine test_qwen4_ref test_mova_kernels"] bash tools/hf_cuda_test.sh
set -euo pipefail
TARGETS=${TARGETS:-"test_qwen4_kernels test_qwen4_engine test_qwen4_ref test_mova_kernels"}
export DEBIAN_FRONTEND=noninteractive
command -v cmake >/dev/null || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq cmake libicu-dev >/dev/null 2>&1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
rm -rf /work && cp -r "$SRC" /work
cmake -S /work -B /work/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null
t0=$(date +%s)
cmake --build /work/build -j "$(nproc)" --target $TARGETS 2>&1 | grep -E "error|warning: " || true
echo "build: $(( $(date +%s) - t0 )) s"
cd /work && ctest --test-dir build -R "^($(echo $TARGETS | tr ' ' '|'))\$" --output-on-failure -V | grep -vE "^[0-9]+: Test command|^UpdateCTestConfiguration|^Parse|^ Add coverage|^Constructing|^Checking test|^Test project|^test [0-9]+$|^ *Start"
