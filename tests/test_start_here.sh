#!/usr/bin/env bash
# tests/test_start_here.sh - start-here.sh gets its engine before it asks for a pack:
#   - a source checkout (start/start-here.sh) builds nslm-chat and nslm-serve, and builds nothing when they are current
#   - a checkout without a compiler, and a release folder without its binaries, download the release archive of their
#     version and check its SHA-256 against the release's SHA256SUMS
# Stubs on PATH: curl (a local release), nvidia-smi and cmake (Linux), xcode-select and xcrun (macOS). The unknown pack
# NOPE stops each run before the pack download, or the RAM check stops it on a machine of less than 64 GB.
# Usage: tests/test_start_here.sh (from any folder).
set -u
repo=$(cd "$(dirname "$0")/.." && pwd)
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
fails=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
past() {   # the run got past the engine step: the pack question, or the RAM check of a small machine (a CI runner)
    grep -q -e "Unknown pack NOPE" -e "needs 64 GB of RAM" "$t/log"
}
check() { if eval "$2"; then ok "$1"; else bad "$1"; echo "     output:"; sed 's/^/     | /' "$t/log"; fi; }

if [ "$(uname -s)" = Darwin ]; then bin=out/bin asset=nslm-v9.9.9-macos-arm64-metal sub=out
else bin=build/bin asset=nslm-v9.9.9-linux-x64-cuda12.8 sub=build; fi
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# the real Makefile has the targets that start-here.sh builds on a Mac
if [ "$(uname -s)" = Darwin ]; then
    check "the Makefile builds nslm-chat, nslm-serve and their metallibs" \
        'make -C "$repo" -n out/bin/nslm-chat out/bin/nslm-serve out/res/kernels_moe.metallib out/res/kernels_qwen4.metallib >/dev/null 2>&1'
fi

# stubs. A good toolchain in good/, a missing one in none/ (it comes first on PATH)
mkdir -p "$t/good" "$t/none" "$t/rel"
cat >"$t/good/nvidia-smi" <<'EOF'
#!/bin/sh
echo "NVIDIA GeForce RTX 4080, 16376, 580.00"
EOF
cat >"$t/good/curl" <<'EOF'
#!/bin/sh
# Files come from $NSLM_TEST_REL: a release file by its name, the pack's tree (to stdout) and its files (pack/PATH).
# -o names the output file, and -C - resumes into it. A pack file logs its -H file and how many run at once.
o= u= c= h=
while [ $# -gt 0 ]; do case $1 in -o) o=$2; shift ;; -C) c=$2; shift ;; -H) h=$2; shift ;; http*) u=$1 ;; esac; shift; done
R=$NSLM_TEST_REL
echo "curl $u" >>"$R/../curl.log"
case $u in
*/api/models/*) cat "$R/tree.json"; exit 0 ;;
*/resolve/*)
    x=${u#*/resolve/} && f=$R/pack/${x#*/}
    case $h in @*) cat "${h#@}" >>"$R/../headers.log" ;; esac
    mkdir -p "$R/../run" && : >"$R/../run/$$" && ls "$R/../run" | wc -l >>"$R/../conc.log"
    sleep 1
    rm -f "$R/../run/$$" ;;
*) f=$R/${u##*/} ;;
esac
[ -f "$f" ] || exit 22
if [ "$c" = - ] && [ -f "$o" ]; then n=$(wc -c <"$o"); tail -c +$((n + 1)) "$f" >>"$o"; else cp "$f" "$o"; fi
EOF
cat >"$t/good/cmake" <<'EOF'
#!/bin/sh
# configure: build/CMakeCache.txt. --build: the targets, as stub binaries
case " $* " in
*" --version "*) echo "cmake version 3.30.0" ;;
*" --build "*) mkdir -p build/bin && for b in nslm-chat nslm-serve; do printf '#!/bin/sh\nexit 2\n' >build/bin/$b; chmod +x build/bin/$b; done
    echo "cmake built" ;;
*) mkdir -p build && : >build/CMakeCache.txt ;;
esac
EOF
printf '#!/bin/sh\necho "Cuda compilation tools, release 12.8"\n' >"$t/good/nvcc"
printf '#!/bin/sh\necho /Library/Developer/CommandLineTools\n' >"$t/good/xcode-select"
printf '#!/bin/sh\nexit 0\n' >"$t/good/xcrun"
printf '#!/bin/sh\nexit 1\n' >"$t/none/xcrun"
printf '#!/bin/sh\nexit 1\n' >"$t/none/cmake"
chmod +x "$t"/good/* "$t"/none/*
export NSLM_TEST_REL="$t/rel"

# a release archive of v9.9.9 with stub binaries, and its SHA256SUMS
mkdir -p "$t/pkg/$asset/$bin" "$t/pkg/$asset/$sub/res"
for b in nslm-chat nslm-serve; do printf '#!/bin/sh\nexit 2\n' >"$t/pkg/$asset/$bin/$b"; chmod +x "$t/pkg/$asset/$bin/$b"; done
echo "a different start-here.sh" >"$t/pkg/$asset/start-here.sh"
(cd "$t/pkg" && tar -czf "$t/rel/$asset.tar.gz" "$asset")
echo "$(sha "$t/rel/$asset.tar.gz")  $asset.tar.gz" >"$t/rel/SHA256SUMS"

run() {   # run DIR PATH_PREFIX: the start script of folder DIR, with the stubs of PATH_PREFIX
    (cd "$t" && PATH="$2:$PATH" bash "$1" NOPE >"$t/log" 2>&1)
    echo $? >"$t/rc"
}

# 1. a checkout without binaries builds them, then passes the engine step
co=$t/checkout
mkdir -p "$co/start"
cp "$repo/start/start-here.sh" "$co/start/"
echo 9.9.9 >"$co/VERSION"
: >"$co/CMakeLists.txt"
cat >"$co/Makefile" <<'EOF'
out/res/%.metallib:
	mkdir -p $(@D) && touch $@
out/bin/%:
	mkdir -p $(@D) && printf '#!/bin/sh\nexit 2\n' >$@ && chmod +x $@
EOF
run "$co/start/start-here.sh" "$t/good"
check "a checkout builds the engine" '[ -x "$co/$bin/nslm-serve" ] && [ -x "$co/$bin/nslm-chat" ]'
check "a checkout says that it builds" 'grep -q "Building nslm-chat and nslm-serve" "$t/log"'
check "a checkout passes the engine step" 'past && ! grep -q "is missing" "$t/log"'
check "a checkout does not download" '[ ! -f "$t/curl.log" ]'

# 2. a checkout with current binaries builds nothing
if [ "$(uname -s)" = Darwin ]; then
    run "$co/start/start-here.sh" "$t/good"
    check "current binaries build nothing" '! grep -q "Building" "$t/log" && past'
fi

# 3. a checkout at the tag of its VERSION downloads the signed release, also with a compiler
if command -v git >/dev/null; then
    rm -rf "$co/out" "$co/build" "$t/curl.log"
    mkdir -p "$co/engine"
    echo "int x;" >"$co/engine/x.c"
    git -C "$co" init -q && git -C "$co" add -A && git -C "$co" -c user.name=t -c user.email=t@t commit -qm t && git -C "$co" tag v9.9.9
    run "$co/start/start-here.sh" "$t/good"
    check "the release's source: the signed release downloads" 'grep -q "releases/download/v9.9.9/$asset.tar.gz" "$t/curl.log" 2>/dev/null'
    check "the release's source: nothing builds" '! grep -q "Building" "$t/log" && [ -x "$co/$bin/nslm-serve" ]'
    check "the release's source: the run passes the engine step" 'past'
    # a change of the engine source after the tag: a build
    rm -rf "$co/out" "$co/build" "$t/curl.log"
    echo "int y;" >>"$co/engine/x.c"
    run "$co/start/start-here.sh" "$t/good"
    check "other engine source: a build, no download" 'grep -q "Building" "$t/log" && [ ! -f "$t/curl.log" ]'
    # a change of the documentation only: still the release
    git -C "$co" checkout -q engine/x.c && rm -rf "$co/out" "$co/build"
    echo doc >"$co/notes.md"
    run "$co/start/start-here.sh" "$t/good"
    check "a change of other files: the signed release downloads" '[ -f "$t/curl.log" ] && ! grep -q "Building" "$t/log"'
    rm -rf "$co/.git" "$co/engine" "$co/notes.md"
fi

# 4. a checkout without a compiler downloads the release of its VERSION
rm -rf "$co/out" "$co/build" "$t/curl.log"
run "$co/start/start-here.sh" "$t/none:$t/good"
check "no compiler: the release of VERSION downloads" 'grep -q "releases/download/v9.9.9/$asset.tar.gz" "$t/curl.log"'
check "no compiler: the binaries unpack into the checkout" '[ -x "$co/$bin/nslm-serve" ] && [ -x "$co/$bin/nslm-chat" ]'
check "no compiler: the archive does not replace start-here.sh" 'cmp -s "$repo/start/start-here.sh" "$co/start/start-here.sh" && [ ! -f "$co/start-here.sh" ]'
check "no compiler: the run passes the engine step" 'past'

# 4. a release folder without its binaries downloads them (the tag is in README.txt)
rf=$t/release
mkdir -p "$rf"
cp "$repo/start/start-here.sh" "$rf/"
echo "NanoSeedLM engine v9.9.9: nslm-chat and nslm-serve, built from https://github.com/mbarnson/NanoSeedLM at abc." >"$rf/README.txt"
rm -f "$t/curl.log"
run "$rf/start-here.sh" "$t/good"
check "a release folder downloads its binaries" '[ -x "$rf/$bin/nslm-serve" ] && [ -x "$rf/$bin/nslm-chat" ]'
check "a release folder keeps its start-here.sh" 'cmp -s "$repo/start/start-here.sh" "$rf/start-here.sh"'
check "a release folder passes the engine step" 'past'

# 5. a wrong SHA-256 stops the run and unpacks nothing
rm -rf "$rf/out" "$rf/build"
echo "0000000000000000000000000000000000000000000000000000000000000000  $asset.tar.gz" >"$t/rel/SHA256SUMS"
run "$rf/start-here.sh" "$t/good"
check "a wrong SHA-256 stops the run" '[ "$(cat "$t/rc")" = 1 ] && grep -q "SHA-256" "$t/log"'
check "a wrong SHA-256 unpacks nothing" '[ ! -e "$rf/$bin/nslm-serve" ]'

# 6. no release for the version: the message tells how to build
rm -f "$t/rel/SHA256SUMS" "$t/rel/$asset.tar.gz"
run "$rf/start-here.sh" "$t/good"
check "no release: the run stops with the build guide" '[ "$(cat "$t/rc")" = 1 ] && grep -q "docs/guides/build.md" "$t/log"'

# 8. the pack download: files run at the same time, hf when it is installed, each file checked by its SHA-256
if [ "$(uname -s)" != Darwin ] && ! command -v python3 >/dev/null; then
    echo "skip the pack download checks (no python3 to read the file list)"
else
    mkdir -p "$t/rel/pack/sub" "$t/nohf" "$t/hfbin" "$t/hfhome"
    printf '#!/bin/sh\nexit 1\n' >"$t/nohf/hf"
    cat >"$t/hfbin/hf" <<'EOF'
#!/bin/sh
# hf version, and hf download REPO FILES... --local-dir D: copies the files from $NSLM_TEST_REL/pack, except
# $NSLM_TEST_HF_SKIP, a file that hf misses (then hf fails)
# (hf 2.x prints its version without the word huggingface_hub: the script asks download --help for --local-dir)
case $1 in version) echo "hf 9.9.9"; exit 0 ;; download) [ "${2:-}" = --help ] && { echo "  --local-dir LOCAL_DIR"; exit 0; }; shift 2 ;; *) exit 2 ;; esac
d= fs=
while [ $# -gt 0 ]; do case $1 in --local-dir) d=$2; shift ;; --revision | --max-workers) shift ;; --*) ;; *) fs="$fs $1" ;; esac; shift; done
echo "hf$fs" >>"$NSLM_TEST_REL/../hf.log"
for f in $fs; do
    [ "$f" = "${NSLM_TEST_HF_SKIP:-}" ] && continue
    mkdir -p "$d/$(dirname "$f")" && cp "$NSLM_TEST_REL/pack/$f" "$d/$f"
done
[ -z "${NSLM_TEST_HF_SKIP:-}" ]
EOF
    chmod +x "$t/nohf/hf" "$t/hfbin/hf"
    echo "filetoken" >"$t/hfhome/token"
    echo '{"model_type": "test"}' >"$t/rel/pack/config.json"
    for f in a.safetensors sub/b.safetensors L0_up_proj.blkt; do head -c 300000 /dev/urandom >"$t/rel/pack/$f"; done
    tree() {   # tree [PATH SHA]: the pack's file list, with a wrong SHA-256 for PATH
        local s f n
        printf '[{"type": "file", "path": ".gitattributes", "size": 5, "oid": "g0"}, {"type": "directory", "path": "sub"}'
        printf ', {"type": "file", "path": "config.json", "size": %d, "oid": "g1"}' "$(wc -c <"$t/rel/pack/config.json")"
        for f in a.safetensors sub/b.safetensors L0_up_proj.blkt; do
            s=$(sha "$t/rel/pack/$f") n=$(($(wc -c <"$t/rel/pack/$f")))
            [ "$f" = "${1:-}" ] && s=$2
            printf ', {"type": "file", "path": "%s", "size": %d, "oid": "g2", "lfs": {"oid": "%s", "size": %d}}' "$f" "$n" "$s" "$n"
        done
        echo ']'
    }
    pk=$t/pk
    mkdir -p "$pk/$bin"
    cp "$repo/start/start-here.sh" "$pk/"
    for b in nslm-chat nslm-serve; do printf '#!/bin/sh\nexit 2\n' >"$pk/$bin/$b"; chmod +x "$pk/$bin/$b"; done
    m=$t/models/tiny
    runp() {   # runp PATH_PREFIX: start-here.sh TINY in $pk. It stops at the chat (the stub nslm-chat)
        rm -f "$t/curl.log" "$t/conc.log" "$t/headers.log" "$t/hf.log"
        (cd "$t" && env -u HF_TOKEN HF_HOME="$t/hfhome" NSLM_PACKS="TINY test/tiny 0123abc 1 0 1 1" PATH="$1:$PATH" \
            bash "$pk/start-here.sh" TINY "$t/models" >"$t/log" 2>&1)
        echo $? >"$t/rc"
    }
    same() { for f in config.json a.safetensors sub/b.safetensors L0_up_proj.blkt; do cmp -s "$t/rel/pack/$f" "$m/$f" || return 1; done; }
    tree >"$t/rel/tree.json"

    runp "$t/nohf:$t/good"
    check "curl: each file downloads and checks out" 'same && [ "$(wc -l <"$m/.nslm-verified")" -eq 4 ] && grep -q "nslm-chat failed" "$t/log"'
    check "curl: files download at the same time" '[ "$(sort -n "$t/conc.log" | tail -1)" -ge 2 ]'
    check "curl: the token of hf auth login" 'grep -q "Authorization: Bearer filetoken" "$t/headers.log"'
    check "curl: no dot file downloads" '[ ! -e "$m/.gitattributes" ]'
    check "curl: a tip tells how to install hf" 'grep -q "pip install -U huggingface_hub" "$t/log"'

    rm -f "$m/a.safetensors" && head -c 1000 "$t/rel/pack/a.safetensors" >"$m/a.safetensors.part"
    runp "$t/nohf:$t/good"
    check "curl: a FILE.part resumes" 'same && [ ! -e "$m/a.safetensors.part" ] && [ "$(grep -c resolve "$t/curl.log")" -eq 1 ]'

    # a folder answer of ~/DIR is the home folder's DIR (read -r does not expand ~)
    mkdir -p "$t/home"
    (cd "$t" && env -u HF_TOKEN HOME="$t/home" HF_HOME="$t/hfhome" NSLM_PACKS="TINY test/tiny 0123abc 1 0 1 1" \
        PATH="$t/nohf:$t/good:$PATH" bash "$pk/start-here.sh" TINY "~/mm" >"$t/log" 2>&1)
    check "a folder ~/DIR is in the home folder" '[ -f "$t/home/mm/tiny/a.safetensors" ] && [ ! -e "$t/~" ] && [ ! -e "$pk/~" ]'

    rm -rf "$m"
    NSLM_TEST_HF_SKIP=sub/b.safetensors runp "$t/hfbin:$t/good"
    check "hf: hf downloads the files" 'grep -q "a.safetensors" "$t/hf.log" 2>/dev/null'
    check "hf: curl downloads the file that hf misses" 'same && [ "$(grep -c resolve "$t/curl.log")" -eq 1 ] && grep -q "resolve/0123abc/sub/b.safetensors" "$t/curl.log"'
    check "hf: no tip with hf and a login" '! grep -q "Tip:" "$t/log"'
    check "hf: each file checks out" '[ "$(wc -l <"$m/.nslm-verified")" -eq 4 ] && grep -q "nslm-chat failed" "$t/log"'

    rm -rf "$m"
    tree a.safetensors 0000000000000000000000000000000000000000000000000000000000000000 >"$t/rel/tree.json"
    runp "$t/nohf:$t/good"
    check "a wrong SHA-256: the file goes, and the run stops" '[ ! -e "$m/a.safetensors" ] && [ ! -e "$m/a.safetensors.part" ] && [ "$(cat "$t/rc")" = 1 ] && grep -q "a.safetensors failed its size or SHA-256 check" "$t/log" && grep -q "Run start-here.sh again" "$t/log"'
    check "a wrong SHA-256: the other files check out" '[ "$(wc -l <"$m/.nslm-verified")" -eq 3 ] && ! grep -q "nslm-chat failed" "$t/log"'
fi

[ "$fails" -eq 0 ] && echo "start-here: all checks passed" || { echo "start-here: $fails checks failed"; exit 1; }
