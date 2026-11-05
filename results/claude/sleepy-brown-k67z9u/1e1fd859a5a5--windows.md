Status: FAIL
Branch: claude/sleepy-brown-k67z9u
Commit: 1e1fd859a5a504cc921e34c81dfe4f8262317448
Platform: windows
Agent: pc-cuda
Time: 2026-11-05T18:37:03Z

MSVC + CUDA 13.1 link FAIL: nslm_engine.lib(qwen4_cuda.c.obj): unresolved external symbol __sync_synchronize referenced in function ct_thread (every exe linking nslm_engine: nslm-chat, nslm-serve, nslm-qwen4-gen, ...). __sync_synchronize is a GCC/Clang builtin; MSVC has none. Fix: atomic_thread_fence(memory_order_seq_cst) from <stdatomic.h> (MSVC 17.5+ has it with /experimental:c11atomics, which win builds may need), or MemoryBarrier() under _WIN32.
