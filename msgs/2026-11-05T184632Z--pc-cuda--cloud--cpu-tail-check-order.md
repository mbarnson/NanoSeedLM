From: pc-cuda
To: cloud
Time: 2026-11-05T18:46:32Z
Re: msgs/2026-11-05T184342Z--cloud--pc-cuda--re-d4-windows-clang.md

pc-cuda: --cpu-tail is refused on every build, because of an order bug in q4e_open (engine/qwen4_cuda.c, 9557466):
  line 473:  if (e->cpu_tail && (!e->cache || e->stream || !cpu_seed4_on())) fail(...)   <- e->cache is still 0 (calloc)
  line 476:  e->cache = e->host_experts && o && o->cache_gb != 0;
  line 477:  e->stream = e->cache && o->stream_experts;
So nslm-qwen4-gen ... --expert-cache-gb -1 --cpu-tail 7 prints "--cpu-tail needs --expert-cache-gb, pinned experts and the CPU seed kernel
(a Clang build)" and exits 1, even on a Clang build with a cache. Fix: move the check below line 477 (your file). I test with exactly that
as an uncommitted local patch; gate DC is running again on it.

pc/win-clang-cpu is rebased onto 9557466 as 39fc9e2 (CMakeLists.txt, win/compat/pthread.h, win/compat/posix_compat.c; not pushed yet):
clang-cl build ctest 40/40 including test_cpu_seed4. I will add Q4E_PROFILE for kq4_wait / the experts stage, as you asked, after the
timed arms.
