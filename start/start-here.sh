#!/usr/bin/env bash
# start/start-here.sh [PACK [MODEL_FOLDER]] - NanoSeedLM's start on Linux (CUDA) and macOS (Metal), from the engine
# folder or a source checkout: check the machine, get the engine, pick a pack, download and check it, chat once, write
# run-PACK.sh and serve. Rerun: resume.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=   # a source checkout keeps this script in start/: the engine folder is the checkout, and the script builds the engine
if [ -f "$here/../start/start-here.sh" ] && [ -f "$here/../Makefile" ] && [ -f "$here/../CMakeLists.txt" ]; then
    here=$(cd "$here/.." && pwd) src=1
fi
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

# the engine (nslm-chat, nslm-serve). The signed release of this version, when it is built from this engine source:
# a release folder, or a checkout at the release's engine source. A checkout with other engine source builds it, and
# downloads the release only when it has no compiler or the build fails.
gh=https://github.com/mbarnson/NanoSeedLM/releases
tag=
if [ -f VERSION ]; then tag=v$(tr -d ' \r\n' <VERSION)
elif [ -f README.txt ]; then tag=$(sed -n '1s/^NanoSeedLM engine \(v[0-9][0-9.]*\):.*/\1/p' README.txt); fi
have() { [ -x "$bin/nslm-serve" ] && [ -x "$bin/nslm-chat" ]; }
dev() { [ -z "$mac" ] || xcode-select -p >/dev/null 2>&1; }   # git and clang on a Mac: else a stub asks to install them
cancc() {   # the compilers of a build
    if [ -n "$mac" ]; then dev && xcrun -sdk macosx -f metal >/dev/null 2>&1
    else cmake --version >/dev/null 2>&1 && { command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ]; }; fi
}
build() {   # make (macOS) or CMake (Linux) builds the two binaries, again when their source changes
    if [ -n "$mac" ]; then
        local tg="out/res/kernels_moe.metallib out/res/kernels_qwen4.metallib out/bin/nslm-chat out/bin/nslm-serve"
        if make -q $tg 2>/dev/null; then return 0; fi
        echo "Building nslm-chat and nslm-serve from this checkout (make)..."
        make -j "$(sysctl -n hw.ncpu)" $tg || return 1
    else
        echo "Building nslm-chat and nslm-serve from this checkout (CMake and CUDA, some minutes the first time)..."
        if [ ! -f build/CMakeCache.txt ]; then   # for this GPU only: a shorter build
            local g=()
            command -v ninja >/dev/null && g=(-G Ninja)
            cmake -S . -B build ${g[@]+"${g[@]}"} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native || return 1
        fi
        cmake --build build --target nslm-chat nslm-serve --parallel || return 1
    fi
}
fetch() {   # the release archive of $tag, checked against the release's SHA256SUMS. Only its out/ or build/ unpacks
    local a sub t want
    [ -n "$tag" ] || { echo "This folder does not name its version (VERSION or README.txt)." >&2; return 1; }
    if [ -n "$mac" ]; then a=nslm-$tag-macos-arm64-metal sub=out
    elif [ "$(uname -m)" = x86_64 ]; then a=nslm-$tag-linux-x64-cuda12.8 sub=build
    else echo "The releases have no Linux engine for $(uname -m)." >&2; return 1; fi
    t=$(mktemp -d)
    echo "Downloading the engine $tag from GitHub ($a.tar.gz)..."
    if ! curl -fsSL --retry 5 -o "$t/$a.tar.gz" "$gh/download/$tag/$a.tar.gz" ||
        ! curl -fsSL --retry 5 -o "$t/SHA256SUMS" "$gh/download/$tag/SHA256SUMS"; then
        rm -rf "$t"; echo "Cannot download the engine $tag from $gh." >&2; return 1
    fi
    want=$(awk -v f="$a.tar.gz" '$2 == f || $2 == "*" f {print $1}' "$t/SHA256SUMS")
    if [ -z "$want" ] || [ "$(sha "$t/$a.tar.gz")" != "$want" ]; then
        rm -rf "$t"; fail "The engine archive $a.tar.gz failed its SHA-256 check. Run start-here.sh again."
    fi
    if ! tar -xzf "$t/$a.tar.gz" -C "$t" "$a/$sub"; then rm -rf "$t"; echo "Cannot unpack $a.tar.gz." >&2; return 1; fi
    mkdir -p "$sub" && cp -R "$t/$a/$sub/." "$sub/"
    rm -rf "$t"
}
same=   # a checkout whose engine source is the source of the release $tag
if [ -n "$src" ] && [ -n "$tag" ] && [ -e .git ] && dev && command -v git >/dev/null &&
    git rev-parse -q --verify "refs/tags/$tag^{commit}" >/dev/null 2>&1 &&
    git diff --quiet "$tag" -- engine harness nslm win Makefile CMakeLists.txt 2>/dev/null; then
    same=1
fi
nobuild="Build the engine (docs/guides/build.md), or download it from $gh."
if [ -n "$src" ] && [ -z "$same" ] && cancc; then   # other engine source than the release
    if ! build; then
        echo "The build failed (see above). The release $tag comes from an earlier source. Downloading it instead."
        fetch || fail "$nobuild"
    fi
elif ! have; then
    [ -n "$src" ] && [ -z "$same" ] && echo "No compiler for a build. The release $tag comes from an earlier source."
    if ! fetch; then
        { [ -n "$src" ] && cancc && build; } || fail "$nobuild"
    fi
fi
have || fail "$bin/nslm-serve is missing. $nobuild"

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
case $root in "~") root=$HOME ;; "~/"*) root=$HOME/${root#"~/"} ;; esac   # read -r keeps ~: else a folder named ~ here
dir=${root:-$here/models}/${repo#*/}
mkdir -p "$dir"

# the files at their pinned revisions, each checked by size and SHA-256. .nslm-verified lists "path size id" for each
# file checked at this revision (id: a large file's SHA-256, git:BLOB for a small one). An earlier revision's line or
# v0.1.0's bare path matches nothing: then a large file of the right size and SHA-256 stays, and any other downloads again.
# the token: HF_TOKEN, or the one that "hf auth login" saved (a signed-in download has higher rate limits). curl reads
# it from a file of mode 600, so it is not in the process list
tok=${HF_TOKEN:-}
tf=${HF_TOKEN_PATH:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/token}
if [ -z "$tok" ] && [ -f "$tf" ]; then tok=$(tr -d ' \r\n' <"$tf"); fi
hdr=$(mktemp)   # 600
trap 'rm -f "$hdr"' EXIT
if [ -n "$tok" ]; then printf 'Authorization: Bearer %s\n' "$tok" >"$hdr"; fi
files() {   # lines "path size id" of the pack
    local t
    t=$(curl -fsSL -H "@$hdr" "https://huggingface.co/api/models/$repo/tree/$rev?recursive=true") ||
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
    f=$dir/$p
    if grep -qxF "$p $s $h" <<<"$old" && [ -f "$f" ] && [ $(( $(wc -c <"$f") )) -eq "$s" ]; then   # checked before
        echo "$p $s $h" >>"$ver"; continue
    fi
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
# The download: NSLM_DL_JOBS files at once (8). hf, when it is installed (NSLM_NO_HF=1: not), then curl for each file
# that remains. Each file is checked by its size and SHA-256 before it joins .nslm-verified.
jobs=${NSLM_DL_JOBS:-8}
say() { if [ -t 1 ]; then printf '\r\033[K%s\n' "$*"; else printf '%s\n' "$*"; fi; }
ok1() {   # ok1 FILE SIZE ID: FILE has the size, and a large file has the SHA-256
    [ -f "$1" ] && [ $(( $(wc -c <"$1") )) -eq "$2" ] && { [ "${3#git:}" != "$3" ] || [ "$(sha "$1")" = "$3" ]; }
}
chk1() {   # chk1 PATH SIZE ID (an xargs job): a file of hf's that checks out joins .nslm-verified. Another goes
    if ok1 "$dir/$1" "$2" "$3"; then echo "$1 $2 $3" >>"$ver"; rm -f "$dir/$1.part"; else rm -f "$dir/$1"; fi
}
get1() {   # get1 PATH SIZE ID (an xargs job): curl into FILE.part (a stalled connection resumes it), then FILE if it checks out
    local d=$dir/$1
    mkdir -p "$(dirname "$d")"
    if ! ok1 "$d.part" "$2" "$3"; then
        if [ -f "$d.part" ] && [ $(( $(wc -c <"$d.part") )) -ge "$2" ]; then rm -f "$d.part"; fi
        curl -fsSL --retry 20 --speed-limit 100000 --speed-time 60 -C - -H "@$hdr" -o "$d.part" \
            "https://huggingface.co/$repo/resolve/$rev/$1" </dev/null || { say "The download of $1 stopped."; return 0; }
        local n=$(( $(wc -c <"$d.part") ))
        if [ "$n" -lt "$2" ]; then say "$1 is not complete."; return 0; fi   # the next run resumes it
        if ! ok1 "$d.part" "$2" "$3"; then rm -f "$d.part"; say "$1 failed its size or SHA-256 check and was removed."; return 0; fi
    fi
    mv -f "$d.part" "$d"
    echo "$1 $2 $3" >>"$ver"
    say "$1"
}
todo() { while read -r p s h; do grep -qxF "$p $s $h" "$ver" || echo "$p $s $h"; done <<<"$list"; }
export dir repo rev ver hdr
export -f say ok1 chk1 get1 sha
todo=$(todo)
hf=   # the hf command of huggingface_hub
if [ -z "${NSLM_NO_HF:-}" ] && hf version 2>/dev/null | grep -q huggingface_hub; then hf=1; fi
if [ -n "$todo" ] && [ -z "$hf" ] && [ -z "${NSLM_NO_HF:-}" ]; then
    echo "Tip: hf can download faster. It uses Hugging Face's Xet transfers and your login. To install it:"
    echo "  pip install -U huggingface_hub && hf auth login"
    echo "Then run start-here.sh again. It continues the download."
elif [ -n "$todo" ] && [ -z "$tok" ]; then
    echo "Tip: \"hf auth login\" (a free Hugging Face account) gives the download higher rate limits."
fi
if [ -n "$todo" ] && [ -n "$hf" ]; then
    echo "Downloading with hf ($(hf version 2>/dev/null | head -1)), $jobs files at once:"
    # shellcheck disable=SC2046   # the paths have no spaces ("path size id" lines)
    hf download "$repo" $(awk '{print $1}' <<<"$todo") --revision "$rev" --local-dir "$dir" --max-workers "$jobs" </dev/null ||
        echo "hf stopped. curl downloads the files that remain."
    echo "Checking the files (SHA-256)..."
    xargs -n 3 -P "$jobs" "$BASH" -c 'chk1 "$@"' _ <<<"$todo" || true
    todo=$(todo)
fi
if [ -n "$todo" ]; then
    n=$(( $(wc -l <<<"$todo") ))
    need=$(while read -r p s _; do   # the bytes that remain: FILE.part files count
        have=0
        [ -f "$dir/$p.part" ] && have=$(( $(wc -c <"$dir/$p.part") ))
        echo $((s > have ? s - have : s))
    done <<<"$todo" | awk '{s += $1} END {printf "%.0f", s}')
    echo "Downloading $n files with curl, $jobs at once:"
    stamp=$dir/.nslm-stamp
    : >"$stamp"
    got() {   # bytes now in FILE.part files and in files written since the start
        find "$dir" -type f ! -path '*/.*' \( -name '*.part' -o -newer "$stamp" \) -exec ls -ln {} + 2>/dev/null |
            awk '{s += $5} END {printf "%.0f", s}'
    }
    (   # a progress line every 2 s on a terminal, else every 60 s
        b0=$(got) t0=$SECONDS st=2
        [ -t 1 ] || st=60
        while sleep "$st"; do
            b=$(( $(got) - b0 )) dt=$(( SECONDS - t0 ))
            l=$(awk -v b="$b" -v t="$dt" -v n="$need" 'BEGIN { r = b / (t > 0 ? t : 1)
                printf "%.1f of %.1f GB, %.0f MB/s", b / 1e9, n / 1e9, r / 1e6
                if (r > 0 && n > b) printf ", %.0f min left", (n - b) / r / 60 }')
            if [ -t 1 ]; then printf '\r\033[K%s' "$l"; else echo "$l"; fi
        done
    ) &
    mon=$!
    xargs -n 3 -P "$jobs" "$BASH" -c 'get1 "$@"' _ <<<"$todo" || true
    kill "$mon" 2>/dev/null || true
    wait "$mon" 2>/dev/null || true
    rm -f "$stamp"
    [ -t 1 ] && printf '\r\033[K'
    todo=$(todo)
    [ -z "$todo" ] || fail "$(( $(wc -l <<<"$todo") )) files did not download or check out (see above). Run start-here.sh again to resume."
fi
rm -f "$hdr"
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
