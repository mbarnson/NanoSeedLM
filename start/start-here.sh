#!/usr/bin/env bash
# start/start-here.sh [PACK [MODEL_FOLDER]] - NanoSeedLM's start on Linux (CUDA) and macOS (Metal), from the engine
# folder: check the machine, pick a pack, download and check it, chat once, write run-PACK.sh and serve. Rerun: resume.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
fail() { printf '\n%s\n' "$*" >&2; exit 1; }

# name, repository, revision, least RAM and VRAM GB of a PC, least memory GB of a Mac (the cards' "For" column),
# download GB. NSLM_PACKS: another table
PACKS=${NSLM_PACKS:-"TCQ4 txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4 4e0d254480f24d444082e935c9b182a4e582730f 96 24 128 128
TCQ3-D4 txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4 e29730a3fea357b710f0b5bd57158c74a82cd7f1 96 16 96 117
TCQ3 txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3 a850ead9cc4e852b9566aaec851213f7e813d193 64 12 96 112"}
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
echo "  TCQ4     0.0415  a GPU of 24 GB or more with 96 to 128 GB of RAM, or a 128 GB Mac (128 GB download)"
echo "  TCQ3-D4  0.0638  a 16 GB GPU with 96 to 128 GB of RAM, or a 96 GB Mac (117 GB download)"
echo "  TCQ3     0.0778  a 12 or 16 GB GPU with 64 GB of RAM, or a 96 GB Mac (112 GB download)"
echo "Each pack carries its PLE table at 8 bits. It stays on the disk, and the page cache holds the rows in use."
pick=${1:-}   # given: no questions
[ -n "$pick" ] || read -r -p "Pack [$rec] " pick
pick=${pick:-$rec}
line=$(awk -v n="$pick" '$1 == n' <<<"$PACKS")
[ -n "$line" ] || fail "Unknown pack $pick. Use TCQ4, TCQ3-D4 or TCQ3."
read -r _ repo rev _ <<<"$line"
root=${2:-}
[ -n "$root" ] || [ -n "${1:-}" ] || read -r -p "Model folder [$here/models] " root
dir=${root:-$here/models}/${repo#*/}
mkdir -p "$dir"

# the files at their pinned revisions, each checked by size and SHA-256. .nslm-verified lists "path size id" for each
# file checked at this revision (id: a large file's SHA-256, git:BLOB for a small one). An earlier revision's line or
# v0.1.0's bare path matches nothing: then a large file of the right size and SHA-256 stays, and any other downloads again.
auth=()   # used as ${auth[@]+"${auth[@]}"}: bash 3.2 (macOS) calls an empty array unbound under set -u
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
files() {   # lines "path size id" of the pack
    local t
    t=$(curl -fsSL ${auth[@]+"${auth[@]}"} "https://huggingface.co/api/models/$repo/tree/$rev?recursive=true") ||
        fail "Cannot list $repo."
    if [ -n "$mac" ]; then   # JavaScript for Automation reads JSON on each Mac. python3 can be a stub that asks for Xcode
        osascript -l JavaScript -e 'function run(a) { return JSON.parse(a[0]).filter(f => f.type == "file" &&
            f.path[0] != ".").map(f => f.path + " " + f.size + " " + (f.lfs ? f.lfs.oid : "git:" + f.oid)).join("\n") }' "$t"
    else
        command -v python3 >/dev/null || fail "start-here.sh needs python3 to read the file list. Install python3."
        python3 -I -c 'import json, sys
for f in json.load(sys.stdin):
    if f["type"] == "file" and not f["path"].startswith("."): print(f["path"], f["size"], (f.get("lfs") or {}).get("oid") or "git:" + f["oid"])' <<<"$t"
    fi
}
list=$(files)
ver=$dir/.nslm-verified
touch "$ver"
old=$(cat "$ver") check=0 need=0 hashed=0
: >"$ver"
while read -r p s h; do
    case $p in /* | *..*) fail "The pack lists an unsafe path: $p" ;; esac
    if grep -qxF "$p $s $h" <<<"$old"; then echo "$p $s $h" >>"$ver"; continue; fi
    f=$dir/$p
    if [ -f "$f" ] && [ $(( $(wc -c <"$f") )) -eq "$s" ] && [ "${h#git:}" = "$h" ]; then   # from an earlier download
        [ "$hashed" -eq 0 ] && echo "Checking the files of an earlier download (SHA-256)..."
        hashed=1
        if [ "$(sha "$f")" = "$h" ]; then echo "$p $s $h" >>"$ver"; continue; fi
    fi
    rm -f "$f"   # an older or broken version: a download starts over, and never writes through a hard link
    check=$((check + 1)) have=0
    [ -f "$f.part" ] && have=$(( $(wc -c <"$f.part") ))
    if [ "$have" -gt "$s" ]; then rm -f "$f.part"; have=0; fi
    need=$((need + s - have))
done <<<"$list"
gone=0   # the files of an earlier download that this revision does not have
while read -r p _; do
    case $p in "" | /* | *..*) continue ;; esac
    [ -f "$dir/$p" ] || continue
    awk -v p="$p" '$1 == p {f = 1} END {exit !f}' <<<"$list" && continue
    gone=$((gone + $(wc -c <"$dir/$p")))
    rm -f "$dir/$p"
done <<<"$old"
[ "$gone" -gt 0 ] && echo "Removed $(awk -v n="$gone" 'BEGIN {printf "%.1f", n / 1e9}') GB of earlier files that this revision of the pack does not have."
free=$(( $(df -Pk "$dir" | awk 'NR == 2 {print $4}') * 1024 ))
[ "$need" -le "$free" ] || fail "The download needs $((need / 1000000000)) GB and the disk has $((free / 1000000000)) GB free."
echo
echo "$check files to download, $(awk -v n="$need" 'BEGIN {printf "%.1f", n / 1e9}') GB into ${root:-$here/models}"
if [ "$need" -gt 1073741824 ] && [ -z "${1:-}" ]; then
    read -r -p "Start? [Y/n] " a
    case $a in [nN]*) echo "Stopped. Run start-here.sh again."; exit 0 ;; esac
elif [ "$need" -gt 1073741824 ]; then
    echo "The download starts in 5 s (Ctrl+C stops it)."; sleep 5   # a folder typo
fi
total=$(( $(wc -l <<<"$list") )) i=0
while read -r p s h; do
    i=$((i + 1))
    grep -qxF "$p $s $h" "$ver" && continue
    dst=$dir/$p
    mkdir -p "$(dirname "$dst")"
    echo "[$i/$total] $p ($(awk -v n="$s" 'BEGIN {if (n < 1e9) printf "%.0f MB", n / 1e6; else printf "%.1f GB", n / 1e9}'))"
    have=0
    [ -f "$dst.part" ] && have=$(( $(wc -c <"$dst.part") ))
    if [ "$have" -ne "$s" ]; then   # into FILE.part: a stalled connection resumes it, and FILE appears once it checks out
        curl -fL --retry 20 --speed-limit 100000 --speed-time 60 -C - -# ${auth[@]+"${auth[@]}"} -o "$dst.part" \
            "https://huggingface.co/$repo/resolve/$rev/$p" </dev/null ||
            fail "The download of $p stopped. Run start-here.sh again to resume."
    fi
    [ $(( $(wc -c <"$dst.part") )) -eq "$s" ] || fail "$p has the wrong size. Run start-here.sh again."
    if [ "${h#git:}" = "$h" ] && [ "$(sha "$dst.part")" != "$h" ]; then rm -f "$dst.part"; fail "$p failed its SHA-256 check and was removed. Run start-here.sh again."; fi
    mv -f "$dst.part" "$dst"
    echo "$p $s $h" >>"$ver"
done <<<"$list"
pline="# the pack's own 8-bit PLE table (Q4E_PLE_GGUF=FILE reads another, as a GGUF)"   # run-PACK.sh's line for the table
dline="# MTP drafts: Q4E_DRAFTS=2 Q4E_DRAFT_MIN_P=0.3 (they break even on an RTX 4080)"   # and for the drafts
[ -n "$mac" ] && dline="export Q4E_DRAFTS=2 Q4E_DRAFT_MIN_P=0.3   # 2 MTP drafts: 32% faster on an M4 Max, the same text"

echo
echo "The pack is ready. One short chat:"
"$bin/nslm-chat" --model "$dir" --no-think --max-tokens 64 "Say hello in one short sentence." || fail "nslm-chat failed. See the message above."
run=$here/run-$pick.sh
cat >"$run" <<EOF
#!/bin/sh
cd "\$(dirname "\$0")" || exit 1
echo "NanoSeedLM $pick: an OpenAI-compatible API on http://127.0.0.1:8080/v1 (Ctrl+C stops it)"
$pline
$dline
exec $bin/nslm-serve --model "$dir" --ctx 32768 --port 8080
EOF
chmod +x "$run"
echo
echo "Wrote $run. Next time, run it to start the server at once."
exec "$run"
