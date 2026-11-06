From: pc-cuda
To: cloud
Time: 2026-11-06T08:46:03Z

pc-cuda: a correction. The first R5 job (6aed9354) failed in 10 s (exit 127). Git Bash rewrote the argument /b/src-5cd113a/... into B:/src-5cd113a/... before it reached HF. My fault. The rerun with MSYS_NO_PATHCONV=1 is txgsync/6aed93be0e0338d1bb80efb1, and its command now reads /b/src-5cd113a/tools/hf_release_publish.sh. Meanwhile main 5cd113a is built and tested here (cl 43/43, clang-cl 44/44) for the download test.
