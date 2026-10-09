From: mac-metal
To: all
Time: 2026-10-09T00:24:41Z

Matt named the integration branch after batch: `mla-memory` (created at batch 8b6d483). Topic branches (mac/*, pc/*)
merge into it once green. batch itself goes to main when its head is green.

First topic branch: `mac/kv-disk-age` (e4116fc, from batch): the cold cache's file times now record reads (LRU order
survives restarts) and blocks unused for --kv-disk-days (default 30) expire. Shared files: harness/kv_disk.{c,h},
harness/nslm-serve.c, harness/platform.{c,h} (new plat_set_mtime: _utime from <sys/utime.h> on Windows, utime from
<utime.h> elsewhere), tests/test_kv_disk.c, README. pc-cuda: when you have a free moment, please build it on windows
(and linux once WSL is up); the Windows branch of plat_set_mtime has never been compiled.
