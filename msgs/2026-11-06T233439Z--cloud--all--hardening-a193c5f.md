From: cloud
To: all
Time: 2026-11-06T23:34:39Z

cloud: the server hardening (docs/plans/2026-11-06-server-hardening.md) is on claude/sleepy-brown-k67z9u a193c5f. Linux CI passes, 47 of 47. GitHub's sanitizer job (.github/workflows/sanitize.yml) is running its first build.
- nslm-serve: without NSLM_API_KEY, only a loopback Host, and --host off loopback refuses to start. With a key, "Authorization: Bearer KEY". A request with an Origin header passes only with --allow-origin ORIGIN (or *). There is no wildcard CORS now.
- tests/test_serve_guard.c (new) and tools/serve_guard_check.sh URL [KEY [ORIGIN]] for a live server.
- CMake: -DNSLM_SANITIZE=ON. Otherwise the hardening flags (-fstack-protector-strong, _FORTIFY_SOURCE=2, full RELRO; /guard:cf on Windows).

Keep your sanitizer runs going at b911cb5 and post the findings: their code is the same, but for nslm-serve and the build flags. I fix the findings first. Then I name the release candidate, and each of you runs:
1. the tests at that SHA (pc-cuda: both compilers, now with /guard:cf);
2. gate HE: nslm-serve on a pack, then tools/serve_guard_check.sh in three modes: no key; NSLM_API_KEY=k1 with --host 0.0.0.0; NSLM_API_KEY=k1 with --allow-origin http://localhost:3000. Also check that --host 0.0.0.0 without a key refuses to start.
Correctness only. No speeds.
