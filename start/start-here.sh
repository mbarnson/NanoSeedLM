#!/usr/bin/env bash
# start/start-here.sh [PACK [MODEL_FOLDER]] - NanoSeedLM's start on Linux (CUDA) and macOS (Metal), from the engine
# folder: check the machine, pick a pack, download and check it, chat once, write run-PACK.sh and serve. Rerun: resume.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
fail() { printf '\n%s\n' "$*" >&2; exit 1; }

# name, repository, revision, least RAM and VRAM GB of a PC, least memory GB of a Mac (the cards' "For" column),
# download GB. NSLM_PACKS: another table
PACKS=${NSLM_PACKS:-"TCQ4 txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4 25339937b55f9efbe384a3f8cb2bc92d86ecfa32 96 24 128 102
TCQ3-D4 txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4 7e5edb2be58071fbe5a5133421ab6c1f2f0de58f 96 16 96 92
TCQ3 txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3 5f2a859927d2215081a7ac5684dbae5da6517104 64 16 96 87"}
# Unsloth's Q8_0 PLE table: the KL on the model cards is with it. NSLM_PLE=pack keeps the pack's 4-bit table
E_REPO=unsloth/Qwen3.8-Flash-Next-GGUF E_REV=d96201194f103c906994f283f504209dc8d2a10c
E_FILE=engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file
fits() { [ $(($1 * 10)) -ge $(($2 * 9)) ]; }   # a 64 GB PC reports about 62 GB: 90% of a size counts as that size

echo "NanoSeedLM: Qwen3.8-Flash-Next on this machine"
echo
if [ "$(uname -s)" = Darwin ]; then
    bin=out/bin
    [ "$(uname -m)" = arm64 ] || fail "NanoSeedLM on macOS needs Apple Silicon."
    [ "$(sw_vers -productVersion | cut -d. -f1)" -ge 15 ] || fail "NanoSeedLM needs macOS 15 or newer."
    ram=$(( $(sysctl -n hw.memsize) / 1073741824 )) vram=0 mac=1   # unified memory: the Mac column
    sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
    echo "Mac: $(sysctl -n machdep.cpu.brand_string), $ram GB."
    if xattr -p com.apple.quarantine "$bin/nslm-serve" >/dev/null 2>&1 &&   # a browser download that is not notarized
        ! spctl --assess --type open --context context:primary-signature "$bin/nslm-serve" 2>/dev/null; then
        echo "macOS does not accept these binaries. Clearing their download quarantine: xattr -dr com.apple.quarantine out"
        xattr -dr com.apple.quarantine out
    fi
else
    bin=build/bin mac=
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
while read -r n _ _ r v m _; do
    [ -n "$mac" ] && r=$m v=0
    [ -z "$rec" ] && fits "$ram" "$r" && fits "$vram" "$v" && rec=$n
done <<<"$PACKS"
if [ -z "$rec" ]; then fits "$ram" 64 || fail "The smallest pack (TCQ3) needs 64 GB of RAM. This machine has $ram GB."; rec=TCQ3; fi
echo
echo "Packs (KL against BF16: lower is closer to the original model):"
echo "  TCQ4     0.0415  a GPU of 24 GB or more with 96 to 128 GB of RAM, or a 128 GB Mac (102 GB download)"
echo "  TCQ3-D4  0.0638  a 16 GB GPU with 96 to 128 GB of RAM, or a 96 GB Mac (92 GB download)"
echo "  TCQ3     0.0778  a 16 GB GPU with 64 GB of RAM, or a 96 GB Mac (87 GB download)"
echo "Each pack runs with Unsloth's Q8_0 PLE table (54.4 GB more), as on the model cards. NSLM_PLE=pack skips it."
pick=${1:-}   # given: no questions
[ -n "$pick" ] || read -r -p "Pack [$rec] " pick
pick=${pick:-$rec}
line=$(awk -v n="$pick" '$1 == n' <<<"$PACKS")
[ -n "$line" ] || fail "Unknown pack $pick. Use TCQ4, TCQ3-D4 or TCQ3."
read -r _ repo rev _ <<<"$line"
root=${2:-}
[ -n "$root" ] || [ -n "${1:-}" ] || read -r -p "Model folder [$here/models] " root
dir=${root:-$here/models}/${repo#*/} edir=${root:-$here/models}/${E_REPO#*/}
ple=${NSLM_PLE:-q8}
if [ "$ple" = q8 ] && [ -z "${1:-}" ]; then
    read -r -p "Also download Unsloth's Q8_0 PLE table (54.4 GB), as on the model card? [Y/n] " a
    case $a in [nN]*) ple=pack ;; esac
fi
[ "$ple" = q8 ] || edir=$dir
mkdir -p "$dir" "$edir"

# the files at their pinned revisions, each checked by size and SHA-256 (verified ones are listed in .nslm-verified)
auth=()   # used as ${auth[@]+"${auth[@]}"}: bash 3.2 (macOS) calls an empty array unbound under set -u
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
src() { if [ "$1" = E ]; then r=$E_REPO v=$E_REV d=$edir; else r=$repo v=$rev d=$dir; fi; }   # P: the pack, E: the table
files() {   # files P|E [SUBFOLDER]: lines "P|E path size sha256" (sha256 - for small files)
    local t
    src "$1"
    t=$(curl -fsSL ${auth[@]+"${auth[@]}"} "https://huggingface.co/api/models/$r/tree/$v${2:+/$2}?recursive=true") ||
        fail "Cannot list $r."
    if [ -n "$mac" ]; then   # JavaScript for Automation reads JSON on each Mac. python3 can be a stub that asks for Xcode
        osascript -l JavaScript -e 'function run(a) { return JSON.parse(a[0]).filter(f => f.type == "file" &&
            f.path[0] != ".").map(f => f.path + " " + f.size + " " + ((f.lfs || {}).oid || "-")).join("\n") }' "$t"
    else
        command -v python3 >/dev/null || fail "start-here.sh needs python3 to read the file list. Install python3."
        python3 -I -c 'import json, sys
for f in json.load(sys.stdin):
    if f["type"] == "file" and not f["path"].startswith("."): print(f["path"], f["size"], (f.get("lfs") or {}).get("oid", "-"))' <<<"$t"
    fi | sed "s/^/$1 /"
}
list=$(files P)
if [ "$ple" = q8 ]; then
    e=$(files E "${E_FILE%/*}" | awk -v f="$E_FILE" '$2 == f')
    [ -n "$e" ] || fail "Cannot find $E_FILE in $E_REPO."
    list+=$'\n'$e
fi
touch "$dir/.nslm-verified" "$edir/.nslm-verified"
check=0 need=0   # files to check; bytes to download (a partial file resumes)
while read -r k p s _; do
    src "$k"
    grep -qxF "$p" "$d/.nslm-verified" && continue
    check=$((check + 1)) have=0
    [ -f "$d/$p" ] && have=$(( $(wc -c <"$d/$p") ))
    if [ "$have" -gt "$s" ]; then rm -f "$d/$p"; have=0; fi
    need=$((need + s - have))
done <<<"$list"
free=$(( $(df -Pk "$dir" | awk 'NR == 2 {print $4}') * 1024 ))
[ "$need" -le "$free" ] || fail "The download needs $((need / 1073741824)) GB and the disk has $((free / 1073741824)) GB free."
echo
echo "$check files to check, $(awk -v n="$need" 'BEGIN {printf "%.1f", n / 1073741824}') GB to download into ${root:-$here/models}"
if [ "$need" -gt 1073741824 ] && [ -z "${1:-}" ]; then
    read -r -p "Start? [Y/n] " a
    case $a in [nN]*) echo "Stopped. Run start-here.sh again."; exit 0 ;; esac
elif [ "$need" -gt 1073741824 ]; then
    echo "The download starts in 5 s (Ctrl+C stops it)."; sleep 5   # a folder typo
fi
total=$(( $(wc -l <<<"$list") )) i=0
while read -r k p s h; do
    i=$((i + 1))
    src "$k"
    grep -qxF "$p" "$d/.nslm-verified" && continue
    dst=$d/$p
    mkdir -p "$(dirname "$dst")"
    echo "[$i/$total] $p ($((s / 1048576)) MB)"
    have=0
    [ -f "$dst" ] && have=$(( $(wc -c <"$dst") ))
    if [ "$have" -ne "$s" ]; then
        curl -fL --retry 5 -C - -# ${auth[@]+"${auth[@]}"} -o "$dst" "https://huggingface.co/$r/resolve/$v/$p" </dev/null ||
            fail "The download of $p stopped. Run start-here.sh again to resume."
    fi
    [ $(( $(wc -c <"$dst") )) -eq "$s" ] || fail "$p has the wrong size. Run start-here.sh again."
    if [ "$h" != - ] && [ "$(sha "$dst")" != "$h" ]; then rm -f "$dst"; fail "$p failed its SHA-256 check and was removed. Run start-here.sh again."; fi
    echo "$p" >>"$d/.nslm-verified"
done <<<"$list"
pline=   # run-PACK.sh's line for the table
[ "$ple" = q8 ] && export Q4E_PLE_GGUF="$edir/$E_FILE" && pline="export Q4E_PLE_GGUF=\"$Q4E_PLE_GGUF\""

echo
echo "The pack is ready. One short chat:"
"$bin/nslm-chat" --model "$dir" --no-think --max-tokens 64 "Say hello in one short sentence." || fail "nslm-chat failed. See the message above."
run=$here/run-$pick.sh
cat >"$run" <<EOF
#!/bin/sh
cd "\$(dirname "\$0")" || exit 1
echo "NanoSeedLM $pick: an OpenAI-compatible API on http://127.0.0.1:8080/v1 (Ctrl+C stops it)"
$pline
exec $bin/nslm-serve --model "$dir" --ctx 32768 --port 8080
EOF
chmod +x "$run"
echo
echo "Wrote $run. Next time, run it to start the server at once."
exec "$run"
