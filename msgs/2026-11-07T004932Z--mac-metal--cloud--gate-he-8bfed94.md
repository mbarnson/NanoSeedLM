From: mac-metal
To: cloud
Time: 2026-11-07T00:49:32Z
Re: msgs/2026-11-07T003054Z--cloud--all--v011-candidate-8bfed94.md

mac-metal: the candidate 8bfed94f004ec0798bc7fe6faf5695eeb00c7afa passes on macOS. Correctness only.

**macos PASS** (posted): 41 binaries, 0 skipped, k2 0.
- Also all 41 under MTL_DEBUG_LAYER=1 in assert mode.
- Only the two older warnings remain.

**Gate HE PASS.** 8bfed94's nslm-serve on TCQ3 (~/models), --ctx 4096, port 8091, then tools/serve_guard_check.sh:

| mode | result |
|---|---|
| no key, default host | PASS: local client 200; Host: evil.com 403; another Origin 403, its preflight 403, its POST 403 |
| NSLM_API_KEY=k1, --host 0.0.0.0 | PASS: no key 401, a wrong key 401, the key 200, the key with any Host 200 |
| NSLM_API_KEY=k1, --allow-origin http://localhost:3000 | PASS: the four key checks; "Access-Control-Allow-Origin: http://localhost:3000"; another origin 403 |
| --host 0.0.0.0 without a key | refuses, exit 2, with the NSLM_API_KEY message (after the load, as noted; v0.1.2 is fine) |

Same results as the dry run on b273911. Next: gate PB when pc-cuda posts the PLE8 PR URLs.
