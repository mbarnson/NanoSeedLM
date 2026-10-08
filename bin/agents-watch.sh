#!/bin/sh
# Report what changed on origin since the last run: branches that moved (their commits), and on the agents branch new
# messages to this agent or to all, new test results and claims changes.  README.md: "Notifications".
#   agents-watch.sh [--once | --until-change | --follow] [--interval SECONDS] [--me AGENT]
set -u
cd "$(dirname "$0")/.." || exit 1
ME=${NSLM_AGENT:-}
MODE=once
INTERVAL=60
while [ $# -gt 0 ]; do
    case $1 in
        --once) MODE=once ;;
        --until-change) MODE=until ;;
        --follow) MODE=follow ;;
        --interval) INTERVAL=$2; shift ;;
        --me) ME=$2; shift ;;
        *) echo "usage: agents-watch.sh [--once|--until-change|--follow] [--interval S] [--me AGENT]" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$ME" ] || { echo "agents-watch: set NSLM_AGENT or pass --me (pc-cuda, mac-metal)" >&2; exit 2; }
STATE=.watch
mkdir -p "$STATE"
OUT=$STATE/report

short() { printf '%.12s' "$1"; }

# What the agents branch added or changed between two commits, as seen by $ME.
report_agents() {   # old new
    git diff --name-only --diff-filter=AM "$1" "$2" -- msgs results claims | while read -r f; do
        case $f in
            msgs/*)
                b=$(basename "$f" .md)
                from=$(printf '%s' "$b" | awk -F-- '{print $2}')
                to=$(printf '%s' "$b" | awk -F-- '{print $3}')
                [ "$from" = "$ME" ] && continue
                if [ "$to" = "$ME" ] || [ "$to" = all ]; then
                    echo "MESSAGE $f"
                    git show "$2:$f" | sed 's/^/  | /'
                else
                    echo "message $f (not for $ME)"
                fi ;;
            results/*)
                st=$(git show "$2:$f" | sed -n 's/^Status: *//p' | head -n 1)
                echo "RESULT $f: ${st:-?}" ;;
            claims/*)
                [ "$f" = "claims/$ME.md" ] && continue
                echo "CLAIMS $f"
                git show "$2:$f" | sed 's/^/  | /' ;;
        esac
    done
}

# One poll: append changes to $OUT; return 0 when something changed, 1 when nothing did, 2 when origin was unreachable.
check() {
    : > "$OUT"
    now=$(git ls-remote --heads origin) || { echo "agents-watch: origin unreachable" >&2; return 2; }
    if [ ! -f "$STATE/refs" ]; then
        printf '%s\n' "$now" > "$STATE/refs"
        echo "agents-watch: first run as $ME; recorded $(printf '%s\n' "$now" | wc -l | tr -d ' ') branches on origin" >&2
        return 1
    fi
    printf '%s\n' "$now" | while read -r sha ref; do
        old=$(awk -v r="$ref" '$2 == r {print $1}' "$STATE/refs")
        [ "$old" = "$sha" ] && continue
        name=${ref#refs/heads/}
        git fetch -q origin "+$ref:refs/remotes/origin/$name" 2>/dev/null
        if [ -z "$old" ]; then
            echo "NEW BRANCH $name at $(git log -1 --format='%h %an: %s' "$sha")"
            continue
        fi
        if git cat-file -e "$old^{commit}" 2>/dev/null && git merge-base --is-ancestor "$old" "$sha"; then
            echo "PUSH $name $(short "$old")..$(short "$sha")"
            git log --format='  %h %an: %s' "$old..$sha" | head -n 30
            [ "$name" = agents ] && report_agents "$old" "$sha"
        else
            echo "PUSH $name $(short "$old")..$(short "$sha") (history rewritten, or the old commit is unknown here)"
            git log --format='  %h %an: %s' -5 "$sha"
        fi
    done >> "$OUT"
    printf '%s\n' "$now" | awk '{print $2}' > "$STATE/now-refs"
    awk '{print $2}' "$STATE/refs" | while read -r ref; do
        grep -qx "$ref" "$STATE/now-refs" || echo "DELETED BRANCH ${ref#refs/heads/}"
    done >> "$OUT"
    printf '%s\n' "$now" > "$STATE/refs"
    # keep this worktree's files current, when that is safe
    if [ "$(git rev-parse --abbrev-ref HEAD)" = agents ] && [ -z "$(git status --porcelain)" ]; then
        git merge -q --ff-only origin/agents 2>/dev/null
    fi
    [ -s "$OUT" ]
}

case $MODE in
    once)
        check && cat "$OUT"
        [ -s "$OUT" ] || echo "agents-watch: nothing new"
        exit 0 ;;
    until)
        while :; do
            if check; then cat "$OUT"; exit 0; fi
            sleep "$INTERVAL"
        done ;;
    follow)
        while :; do
            if check; then echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ)"; cat "$OUT"; fi
            sleep "$INTERVAL"
        done ;;
esac
