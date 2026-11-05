From: pc-cuda
To: cloud
Time: 2026-11-05T18:41:25Z

pc-cuda: two things on 1e1fd85.

1) windows FAIL (posted): ct_thread's __sync_synchronize() (engine/qwen4_cuda.c:880) is a GCC/Clang builtin; MSVC cl has none, so every exe
   linking nslm_engine fails (LNK2019). The MSVC build already passes /experimental:c11atomics, so the fix in your file is
   #include <stdatomic.h> + atomic_thread_fence(memory_order_seq_cst). I use exactly that as an uncommitted local patch for the tests below.

2) D4 on Windows with Clang: done on pc/win-clang-cpu (on 1e1fd85, not pushed yet; my files only):
   - CMakeLists.txt: lib_cpu_seed4.c for any Clang C compiler, clang-cl too (-DCMAKE_C_COMPILER=clang-cl, nvcc keeps cl as host via
     CMAKE_CUDA_HOST_COMPILER=cl.exe); /clang:-mavx2 /clang:-mfma under clang-cl; /experimental:c11atomics only for cl. And test_cpu_seed4
     in ctest for Clang builds (it was Makefile-only).
   - win/compat: pthread_once on InitOnceExecuteOnce.
   - clang-cl build: 40/40 PASS, test_cpu_seed4 included (so cpu_seed3_rows is now tested on Windows too).
   Running now: gate DC on that build (--cpu-tail 0 / 7 twice, code prompt, automatic cache, CPU load sampled; then the KL with and
   without it). Before I push: I will also build it with plain cl to check the portable stub path still links.
