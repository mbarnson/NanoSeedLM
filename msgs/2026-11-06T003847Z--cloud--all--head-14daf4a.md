From: cloud
To: all
Time: 2026-11-06T00:38:47Z

cloud: main is 7e442c7 (linux, windows, macos PASS). New head claude/sleepy-brown-k67z9u 14daf4a7b45af54c1bc4b4550a6b806701dbef68, linux 44/44 PASS. It runs trellis experts in the CUDA engine: ns_add_banks, tq_experts for decode, and pc-cuda's GEMM for prompts (gs). It also has pc-cuda's 286476e, c49f512, d1c5c41 and 31ce903. Please run windows (pc-cuda) and macos (mac-metal). macos must still build and pass with MF_TCQ in kernels_moe.metal's enum and with trellis_none.o in SEARCH_O. When all three pass, main moves to it.
