#!/usr/bin/env bash
# tests/test_start_here.sh - start-here.sh gets its engine before it asks for a pack:
#   - a source checkout (start/start-here.sh) builds nslm-chat and nslm-serve, and builds nothing when they are current
#   - a checkout without a compiler, and a release folder without its binaries, download the release archive of their
#     version and check its SHA-256 against the release's SHA256SUMS
# Stubs on PATH: curl (a local release), nvidia-smi and cmake (Linux), xcode-select and xcrun (macOS). The unknown pack
# NOPE stops each run before the pack download. Usage: tests/test_start_here.sh (from any folder).
set -u
repo=$(cd "$(dirname "$0")/.." && pwd)
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
fails=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
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
# the last argument is the URL. -o names the output file. Files come from $NSLM_TEST_REL
o= u=
while [ $# -gt 0 ]; do case $1 in -o) o=$2; shift ;; http*) u=$1 ;; esac; shift; done
echo "curl $u" >>"$NSLM_TEST_REL/../curl.log"
f=$NSLM_TEST_REL/${u##*/}
[ -f "$f" ] || exit 22
cp "$f" "$o"
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

# 1. a checkout without binaries builds them, then reaches the pack question
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
check "a checkout reaches the pack question" 'grep -q "Unknown pack NOPE" "$t/log" && ! grep -q "is missing" "$t/log"'
check "a checkout does not download" '[ ! -f "$t/curl.log" ]'

# 2. a checkout with current binaries builds nothing
if [ "$(uname -s)" = Darwin ]; then
    run "$co/start/start-here.sh" "$t/good"
    check "current binaries build nothing" '! grep -q "Building" "$t/log" && grep -q "Unknown pack NOPE" "$t/log"'
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
    check "the release's source: the run reaches the pack question" 'grep -q "Unknown pack NOPE" "$t/log"'
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
check "no compiler: the run reaches the pack question" 'grep -q "Unknown pack NOPE" "$t/log"'

# 4. a release folder without its binaries downloads them (the tag is in README.txt)
rf=$t/release
mkdir -p "$rf"
cp "$repo/start/start-here.sh" "$rf/"
echo "NanoSeedLM engine v9.9.9: nslm-chat and nslm-serve, built from https://github.com/mbarnson/NanoSeedLM at abc." >"$rf/README.txt"
rm -f "$t/curl.log"
run "$rf/start-here.sh" "$t/good"
check "a release folder downloads its binaries" '[ -x "$rf/$bin/nslm-serve" ] && [ -x "$rf/$bin/nslm-chat" ]'
check "a release folder keeps its start-here.sh" 'cmp -s "$repo/start/start-here.sh" "$rf/start-here.sh"'
check "a release folder reaches the pack question" 'grep -q "Unknown pack NOPE" "$t/log"'

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

[ "$fails" -eq 0 ] && echo "start-here: all checks passed" || { echo "start-here: $fails checks failed"; exit 1; }
