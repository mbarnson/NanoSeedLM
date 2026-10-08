#!/bin/sh
# Write a message, a test result or this agent's claims (the text comes from stdin), commit it, and push it to
# origin/agents, rebasing and retrying when the other agent pushed first.  README.md: "Messages", "Green".
#   agents-post.sh msg    --to AGENT|all|matt --slug SLUG [--re MSGFILE]
#   agents-post.sh result --branch BRANCH --sha SHA --platform macos|windows|linux --status PASS|FAIL
#   agents-post.sh claim                       (stdin replaces claims/$NSLM_AGENT.md; empty stdin: no claims)
# Common: --me AGENT (default $NSLM_AGENT).  $AGENTS_TRAILER, when set, is appended to the commit message.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 1 ] || { sed -n '2,7p' "$0" >&2; exit 2; }
SUB=$1; shift
ME=${NSLM_AGENT:-}
TO= SLUG= RE= BRANCH= SHA= PLATFORM= STATUS=
while [ $# -gt 0 ]; do
    case $1 in
        --me) ME=$2 ;;
        --to) TO=$2 ;;
        --slug) SLUG=$2 ;;
        --re) RE=$2 ;;
        --branch) BRANCH=$2 ;;
        --sha) SHA=$2 ;;
        --platform) PLATFORM=$2 ;;
        --status) STATUS=$2 ;;
        *) echo "agents-post: unknown option $1" >&2; exit 2 ;;
    esac
    shift 2
done
die() { echo "agents-post: $*" >&2; exit 2; }
[ -n "$ME" ] || die "set NSLM_AGENT or pass --me"
[ "$(git rev-parse --abbrev-ref HEAD)" = agents ] || die "run it in the agents worktree"
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
STAMP=$(date -u +%Y-%m-%dT%H%M%SZ)
BODY=$(cat)

case $SUB in
    msg)
        [ -n "$TO" ] && [ -n "$SLUG" ] || die "msg needs --to and --slug"
        F="msgs/$STAMP--$ME--$TO--$SLUG.md"
        mkdir -p msgs
        { echo "From: $ME"; echo "To: $TO"; echo "Time: $NOW"; [ -n "$RE" ] && echo "Re: $RE"; echo; printf '%s\n' "$BODY"; } > "$F"
        SUBJ="$ME -> $TO: $SLUG" ;;
    result)
        [ -n "$BRANCH" ] && [ -n "$SHA" ] && [ -n "$PLATFORM" ] && [ -n "$STATUS" ] || die "result needs --branch --sha --platform --status"
        case $STATUS in PASS|FAIL) ;; *) die "--status is PASS or FAIL" ;; esac
        FULL=$(git rev-parse --verify -q "$SHA^{commit}" 2>/dev/null || echo "$SHA")
        F="results/$BRANCH/$(printf '%.12s' "$FULL")--$PLATFORM.md"
        mkdir -p "$(dirname "$F")"
        { echo "Status: $STATUS"; echo "Branch: $BRANCH"; echo "Commit: $FULL"; echo "Platform: $PLATFORM";
          echo "Agent: $ME"; echo "Time: $NOW"; echo; printf '%s\n' "$BODY"; } > "$F"
        SUBJ="$ME: $PLATFORM $STATUS on $BRANCH $(printf '%.12s' "$FULL")" ;;
    claim)
        F="claims/$ME.md"
        mkdir -p claims
        { echo "# $ME claims (updated $NOW)"; echo; if [ -n "$BODY" ]; then printf '%s\n' "$BODY"; else echo "(none)"; fi; } > "$F"
        SUBJ="$ME: claims" ;;
    *) die "first argument is msg, result or claim" ;;
esac

git add -- "$F" || exit 1
if [ -n "${AGENTS_TRAILER:-}" ]; then
    git commit -q -m "$SUBJ" -m "$AGENTS_TRAILER" || exit 1
else
    git commit -q -m "$SUBJ" || exit 1
fi
i=0
while [ $i -lt 5 ]; do
    if git pull -q --rebase origin agents 2>/dev/null || [ $i -eq 0 ]; then
        git push -q origin HEAD:agents && { echo "agents-post: pushed $F"; exit 0; }
    fi
    i=$((i + 1))
    sleep 3
done
echo "agents-post: committed $F but could not push; run: git pull --rebase origin agents && git push origin HEAD:agents" >&2
exit 1
