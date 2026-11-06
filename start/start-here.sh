#!/usr/bin/env bash
# start/start-here.sh [PACK [MODEL_FOLDER]] - NanoSeedLM's start on Linux (CUDA) and macOS (Metal), from the engine
# folder: check the machine, pick a pack, download and check it, chat once, write run-PACK.sh and serve. Rerun: resume.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
fail() { printf '\n%s\n' "$*" >&2; exit 1; }

# name, repository, revision, least RAM GB, least VRAM GB (the model cards' "For" column), download GB; NSLM_PACKS: another table
PACKS=${NSLM_PACKS:-"TQ4 txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ4 25339937b55f9efbe384a3f8cb2bc92d86ecfa32 96 24 102
TQ3-D4 txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4 7e5edb2be58071fbe5a5133421ab6c1f2f0de58f 96 16 92
TQ3 txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3 5f2a859927d2215081a7ac5684dbae5da6517104 64 16 87"}

echo "NanoSeedLM: Qwen3.8-Flash-Next on this machine"
echo
if [ "$(uname -s)" = Darwin ]; then
    bin=out/bin
    [ "$(uname -m)" = arm64 ] || fail "NanoSeedLM on macOS needs Apple Silicon."
    [ "$(sw_vers -productVersion | cut -d. -f1)" -ge 15 ] || fail "NanoSeedLM needs macOS 15 or newer."
    ram=$(( $(sysctl -n hw.memsize) / 1073741824 )) vram=$ram   # unified memory
    sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
    echo "Mac: $(sysctl -n machdep.cpu.brand_string), $ram GB."
else
    bin=build/bin
    command -v nvidia-smi >/dev/null || fail "No NVIDIA driver found. Install driver 570 or newer: https://www.nvidia.com/drivers"
    IFS=, read -r gpu vmib drv < <(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1)
    vram=$(( ${vmib// /} / 1024 )) drv=${drv// /}
    ram=$(( $(awk '/^MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
    sha() { sha256sum "$1" | cut -d' ' -f1; }
    echo "GPU: $gpu, $vram GB, driver $drv. RAM: $ram GB."
    [ "${drv%%.*}" -ge 570 ] || fail "The driver is older than 570 (CUDA 12.8). Update it: https://www.nvidia.com/drivers"
fi
[ -x "$bin/nslm-serve" ] || fail "$bin/nslm-serve is missing. Unpack the whole engine archive, then run start-here.sh in it."

rec=
while read -r n _ _ r v _; do [ -z "$rec" ] && [ "$ram" -ge "$r" ] && [ "$vram" -ge "$v" ] && rec=$n; done <<<"$PACKS"
if [ -z "$rec" ]; then [ "$ram" -ge 64 ] || fail "The smallest pack (TQ3) needs 64 GB of RAM. This machine has $ram GB."; rec=TQ3; fi
echo
echo "Packs (KL against BF16: lower is closer to the original model):"
echo "  TQ4     0.0415  a GPU of 24 GB or more with 96 to 128 GB of RAM, or a 128 GB Mac (102 GB download)"
echo "  TQ3-D4  0.0638  a 16 GB GPU with 96 to 128 GB of RAM, or a 96 GB Mac (92 GB download)"
echo "  TQ3     0.0778  a 16 GB GPU with 64 GB of RAM, or a 96 GB Mac (87 GB download)"
pick=${1:-}   # given: no questions
[ -n "$pick" ] || read -r -p "Pack [$rec] " pick
pick=${pick:-$rec}
line=$(awk -v n="$pick" '$1 == n' <<<"$PACKS")
[ -n "$line" ] || fail "Unknown pack $pick. Use TQ4, TQ3-D4 or TQ3."
read -r _ repo rev _ <<<"$line"
root=${2:-}
[ -n "$root" ] || [ -n "${1:-}" ] || read -r -p "Model folder [$here/models] " root
dir=${root:-$here/models}/${repo#*/}
mkdir -p "$dir"

# the pack's files at the pinned revision, each checked by size and SHA-256 (verified ones are listed in .nslm-verified)
auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
tree=$(curl -fsSL "${auth[@]}" "https://huggingface.co/api/models/$repo/tree/$rev?recursive=true") || fail "Cannot list $repo."
if command -v python3 >/dev/null; then
    list=$(python3 -I -c 'import json, sys
for f in json.load(sys.stdin):
    if f["type"] == "file" and not f["path"].startswith("."): print(f["path"], f["size"], (f.get("lfs") or {}).get("oid", "-"))' <<<"$tree")
else   # macOS without python3: plutil reads JSON
    printf '%s' "$tree" >"$dir/.tree.json" list= k=0
    while t=$(plutil -extract "$k.type" raw -o - "$dir/.tree.json" 2>/dev/null); do
        p=$(plutil -extract "$k.path" raw -o - "$dir/.tree.json")
        if [ "$t" = file ] && [ "${p#.}" = "$p" ]; then
            list+="$p $(plutil -extract "$k.size" raw -o - "$dir/.tree.json") $(plutil -extract "$k.lfs.oid" raw -o - "$dir/.tree.json" 2>/dev/null || echo -)"$'\n'
        fi
        k=$((k + 1))
    done
fi
okf=$dir/.nslm-verified
touch "$okf"
check=0 need=0   # files to check; bytes to download (a partial file resumes)
while read -r p s _; do
    grep -qxF "$p" "$okf" && continue
    check=$((check + 1)) have=0
    [ -f "$dir/$p" ] && have=$(( $(wc -c <"$dir/$p") ))
    if [ "$have" -gt "$s" ]; then rm -f "$dir/$p"; have=0; fi
    need=$((need + s - have))
done <<<"$list"
free=$(( $(df -Pk "$dir" | awk 'NR == 2 {print $4}') * 1024 ))
[ "$need" -le "$free" ] || fail "The download needs $((need / 1073741824)) GB and the disk has $((free / 1073741824)) GB free."
echo
echo "$repo: $check files to check, $(awk -v n="$need" 'BEGIN {printf "%.1f", n / 1073741824}') GB to download into $dir"
if [ "$need" -gt 1073741824 ] && [ -z "${1:-}" ]; then
    read -r -p "Start? [Y/n] " a
    case $a in [nN]*) echo "Stopped. Run start-here.sh again."; exit 0 ;; esac
elif [ "$need" -gt 1073741824 ]; then
    echo "The download starts in 5 s (Ctrl+C stops it)."; sleep 5   # a folder typo
fi
total=$(wc -l <<<"$list") i=0
while read -r p s h; do
    i=$((i + 1))
    grep -qxF "$p" "$okf" && continue
    dst=$dir/$p
    mkdir -p "$(dirname "$dst")"
    echo "[$i/$total] $p ($((s / 1048576)) MB)"
    have=0
    [ -f "$dst" ] && have=$(( $(wc -c <"$dst") ))
    if [ "$have" -ne "$s" ]; then
        curl -fL --retry 5 -C - -# "${auth[@]}" -o "$dst" "https://huggingface.co/$repo/resolve/$rev/$p" </dev/null ||
            fail "The download of $p stopped. Run start-here.sh again to resume."
    fi
    [ $(( $(wc -c <"$dst") )) -eq "$s" ] || fail "$p has the wrong size. Run start-here.sh again."
    if [ "$h" != - ] && [ "$(sha "$dst")" != "$h" ]; then rm -f "$dst"; fail "$p failed its SHA-256 check and was removed. Run start-here.sh again."; fi
    echo "$p" >>"$okf"
done <<<"$list"

echo
echo "The pack is ready. One short chat:"
"$bin/nslm-chat" --model "$dir" --no-think --max-tokens 64 "Say hello in one short sentence." || fail "nslm-chat failed. See the message above."
run=$here/run-$pick.sh
cat >"$run" <<EOF
#!/bin/sh
cd "\$(dirname "\$0")" || exit 1
echo "NanoSeedLM $pick: an OpenAI-compatible API on http://127.0.0.1:8080/v1 (Ctrl+C stops it)"
exec $bin/nslm-serve --model "$dir" --ctx 32768 --port 8080
EOF
chmod +x "$run"
echo
echo "Wrote $run. Next time, run it to start the server at once."
exec "$run"
