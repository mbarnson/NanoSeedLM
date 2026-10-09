#!/bin/sh
# Build a NanoSeedLM commit on Linux with CUDA and run ctest, on a Hugging Face Jobs GPU (about 2 minutes, ~$0.10).
#   sh bin/hf-linux-ci.sh REV [FLAVOR]        (FLAVOR default rtx-pro-6000; needs `hf auth login`)
# Prints the job id; the log ends with ctest's summary.  Post it with agents-post.sh result --platform linux.
set -eu
REV=${1:?usage: hf-linux-ci.sh REV [FLAVOR]}
FLAVOR=${2:-rtx-pro-6000}
JOB='set -e
export DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
apt-get update -qq && apt-get install -y -qq cmake ninja-build git libicu-dev pkg-config >/dev/null
nvidia-smi --query-gpu=name --format=csv,noheader
cd /tmp && git clone -q https://github.com/mbarnson/NanoSeedLM && cd NanoSeedLM && git checkout -q "$NSLM_REV" && git log --oneline -1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >/tmp/cmake.log 2>&1 || { tail -30 /tmp/cmake.log; exit 1; }
cmake --build build >/tmp/build.log 2>&1 || { grep -E "error|FAILED" /tmp/build.log | head -40; exit 1; }
echo "warnings: $(grep -c warning /tmp/build.log || true)"
ctest --test-dir build --output-on-failure 2>&1 | tail -25'
hf jobs run --detach --flavor "$FLAVOR" --timeout 30m --name "nslm-linux-ci-$REV" -e NSLM_REV="$REV" \
    nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c "$JOB"
echo "then: hf jobs wait <id> && hf jobs logs <id>"
