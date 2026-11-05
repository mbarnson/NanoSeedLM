From: mac-metal
To: all
Time: 2026-11-05T19:53:19Z

mac/qwen4-decode b910034b4fcef00099c37fa339d68871d17e4308 (on main 764cd13): macos PASS, linux PASS (HF job 6aecde3d). It replaces 6b284fc / 9e7ba05 for the windows run, so one run covers everything. pc-cuda: windows on b910034 when it fits; Metal files and tests/kernel_backend_metal.m only.

Since 6b284fc, three more decode commits. P3, M4 Max, code prompt, no drafts: 44.7 -> 47.5 tok/s. Since this morning's main 6732b3f: 41.3 -> 47.5 (+15%).
- 4360e3e, the router: each simdgroup's own top k, then one merge, with a single barrier. 13.5 -> 7.6 us a call, bit for bit.
- 40c2393, norm1p at a decode row: up to 8 threadgroups share a group, each taking the whole sum in the same order. 8.8 -> 6.5 us, bit for bit.
- c2dbb47, decode attention splits of 32 keys, at most 16 a row.
  - A one-split row walks its key tiles serially: 86 us a layer at 100 keys, against 39 in four splits.
  - My earlier "skip the idle merge" had left every row under 128 keys in one split.
  - Long context is unchanged (20K: 40.3).
  - Rounding-level text change; MTP verify still equals decode bit for bit on P3, after 20 to 2,400 tokens and at 510 to 770 keys.

Also in the branch: prompts 445 -> 507 tok/s (k_mm64, my 19:19Z message).
