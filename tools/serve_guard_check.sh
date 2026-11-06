#!/bin/sh
# tools/serve_guard_check.sh URL [KEY [ORIGIN]] - nslm-serve's access rules against a running server, one line a check.
# No KEY: the default server (loopback, no key). KEY: a server with NSLM_API_KEY=KEY. ORIGIN: and --allow-origin ORIGIN.
U=${1:?usage: serve_guard_check.sh http://127.0.0.1:8080 [KEY [ORIGIN]]}
K=${2:-} O=${3:-} fails=0
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
check() {   # check NAME WANT CURL_ARGS...
    n=$1 w=$2
    shift 2
    g=$(code "$@")
    if [ "$g" = "$w" ]; then echo "PASS $n ($g)"; else echo "FAIL $n: $g, want $w"; fails=$((fails + 1)); fi
}
A=
[ -n "$K" ] && A="Authorization: Bearer $K"
if [ -z "$K" ]; then
    check "a local client" 200 "$U/v1/models"
    check "DNS rebinding (Host: evil.com)" 403 -H "Host: evil.com" "$U/v1/models"
    check "a page of another origin" 403 -H "Origin: http://evil.com" "$U/v1/models"
    check "its preflight" 403 -X OPTIONS -H "Origin: http://evil.com" "$U/v1/chat/completions"
    check "its POST" 403 -H "Origin: http://evil.com" -H "Content-Type: text/plain" -d '{}' "$U/v1/chat/completions"
else
    check "no key" 401 "$U/v1/models"
    check "a wrong key" 401 -H "Authorization: Bearer x$K" "$U/v1/models"
    check "the key" 200 -H "$A" "$U/v1/models"
    check "the key, any Host" 200 -H "$A" -H "Host: nslm.example" "$U/v1/models"
fi
if [ -n "$O" ]; then
    h=$(curl -s -D - -o /dev/null -H "Origin: $O" ${A:+-H "$A"} "$U/v1/models" | tr -d '\r' | grep -i '^access-control-allow-origin:')
    case $h in *"$O"* | *"*") echo "PASS the allowed origin gets CORS ($h)" ;; *) echo "FAIL no CORS line for $O"; fails=$((fails + 1)) ;; esac
    check "another origin" 403 -H "Origin: http://evil.com" ${A:+-H "$A"} "$U/v1/models"
fi
[ "$fails" = 0 ] && echo "serve_guard_check: PASS" || { echo "serve_guard_check: FAIL ($fails)"; exit 1; }
