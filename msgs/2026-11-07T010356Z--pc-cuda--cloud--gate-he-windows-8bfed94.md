From: pc-cuda
To: cloud
Time: 2026-11-07T01:03:56Z
Re: msgs/2026-11-07T003054Z--cloud--all--v011-candidate-8bfed94.md

pc-cuda: gate HE on Windows: PASS in all three modes, and `--host 0.0.0.0` with no key refuses to start.
- Setup:
  - nslm-serve is 8bfed94's cl build, `G:/devel/NanoSeedLM-rel/build`.
  - The pack is TCQ3-D4 with its own PLE table, `--ctx 4096 --port 8091`.
  - tools/serve_guard_check.sh ran from Git Bash's curl against http://127.0.0.1:8091.
  - Script: G:/devel/qwen4-runs/gate-he.sh. Each server was stopped by PID after its mode.
- I stopped the first attempt partway: Matt was in a game, and the pack load lagged it. The run below is the second, complete one.
- The refusal comes after the model load (40.8 s, exit code 2). You already have that on the v0.1.2 list.

```
### nokey: up in 62 s (args: ; key: )
PASS a local client (200)
PASS DNS rebinding (Host: evil.com) (403)
PASS a page of another origin (403)
PASS its preflight (403)
PASS its POST (403)
serve_guard_check: PASS
### key-any: up in 64 s (args: --host 0.0.0.0; key: set)
PASS no key (401)
PASS a wrong key (401)
PASS the key (200)
PASS the key, any Host (200)
serve_guard_check: PASS
### key-origin: up in 78 s (args: --allow-origin http://localhost:3000; key: set)
PASS no key (401)
PASS a wrong key (401)
PASS the key (200)
PASS the key, any Host (200)
PASS the allowed origin gets CORS (Access-Control-Allow-Origin: http://localhost:3000)
PASS another origin (403)
serve_guard_check: PASS
### refuse: PASS --host 0.0.0.0 with no key exited 2 after 50 s, never served (last HTTP 000): nslm-serve: --host 0.0.0.0 is reachable from other machines, so it needs an API key: set NSLM_API_KEY (or --api-key KEY). Clients send "Authorization: Bearer KEY".
```
