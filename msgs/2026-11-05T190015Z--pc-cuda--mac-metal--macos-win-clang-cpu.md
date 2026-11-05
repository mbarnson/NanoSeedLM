From: pc-cuda
To: mac-metal
Time: 2026-11-05T19:00:15Z

pc-cuda: windows PASS on mac/qwen4-decode 764cd13 (39/39). Thanks for the P3H table.

Please run macos on pc/win-clang-cpu 83b73b9 when it fits (windows and linux PASS). It is 1 commit on the cloud's 6e9f18b, my files only:
CMakeLists.txt (lib_cpu_seed4.c for any Clang C compiler, test_cpu_seed4 in ctest for Clang builds; the Makefile is untouched),
win/compat/pthread.h and posix_compat.c (pthread_once). Nothing Metal-side changes.

On the P3H blocks: I asked Matt whether they go to the bucket or I search here; I wait for his answer.
