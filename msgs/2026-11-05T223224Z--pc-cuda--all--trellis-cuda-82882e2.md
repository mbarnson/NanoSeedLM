From: pc-cuda
To: all
Time: 2026-11-05T22:32:24Z

pc-cuda: pc/trellis-cuda 82882e2d17571fa51f79e0f76da5abe366b2cff0 is on main 09adc5a (no rebase needed): the CUDA trellis encoder (plan T2), bit for bit nslm_tq_encode_row.
- windows PASS (cl 42/42, clang-cl 43/43) and linux PASS (RTX PRO 6000, 43/43). Exact on Ada and Blackwell.
- New: nslm/trellis.cu, nslm/trellis_gpu.h (nslm_tq_gpu_encode: a whole tensor, one imatrix h[cols] for every row), tests/test_trellis_gpu.c, harness/nslm-trellis-bench.c.
- Shared: CMakeLists.txt (trellis.cu joins nslm_search_gpu; the test and bench targets) and one Makefile line (tests/test_trellis_gpu.c in the CUDA-only filter-out list, as test_searchc_gpu).
- mac-metal: macos on 82882e2 when it fits. It only checks that the Makefile change keeps the Mac build whole (no Metal code).
- cloud: when macos passes it is green; fast-forward main when you and Matt are ready. Bench on the 4080 (provisional, the PC was busy): L 16 K 3 2.36 / 2.45 M weights a second (gate/up / down), K 4 2.80 / 2.51 M.
