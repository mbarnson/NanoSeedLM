# mac-metal claims (updated 2026-10-09T00:20:53Z)

- **`batch` at 8b6d483:** macOS PASS posted; Matt wants to merge it to main once windows/linux are green.
- **Now, on `mac/kv-disk-age` (from batch 8b6d483):** the cold cache's eviction: a read refreshes a block's age
  (so least-recently-used survives restarts), and blocks unused for --kv-disk-days (default 30) expire. Shared files:
  `harness/kv_disk.{c,h}`, `harness/nslm-serve.c`, `tests/test_kv_disk.c`, README. No engine files.
- **Next (after the merge):** Metal MLA decode kernel rework; KV page hand-back when a slot is reused for a shorter
  conversation (Metal); then the MLA projection pack (shared packer + both engines: I'll message before touching it).
