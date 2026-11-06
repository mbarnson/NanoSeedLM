// engine/qwen4_eng.h - the qwen4 (qwen4_exp) CUDA engine: an MLX model folder on the GPU, tokens in, logits out.
// Routed experts in mapped host memory (or VRAM), the rest in VRAM; prompts run in chunks of Q4EOpts.rows tokens.
#pragma once
#include <stdint.h>

#define Q4E_MAX_ROWS 4096    // default rows of one forward without logits (a chunk): about 0.25 MB of VRAM a row. More rows
                             // fill the GEMM tiles: each expert gets about rows * top_k / experts of them
#define Q4E_BIG_ROWS 16384   // the default on GPUs of 64 GB or more (RTX PRO 6000, 29,120 tokens: +17% for about 3 GB)
#define Q4E_LOGIT_ROWS 128   // rows of one forward with logits (the logits take vocab floats a row)
#define Q4E_PREFETCH_PICKS 64 // forwards of this many router picks an expert copy the next layer's experts beside this
                             // layer's compute (fewer: mostly unused experts; it pays from about 3,000 rows here)
#define Q4E_PASS_ROWS 32768  // most prompt rows of one layer-major pass (the context, rounded up to chunks): each layer's
                             // missing experts cross PCIe once a pass

#include "qwen4_cfg.h"

typedef struct Q4Eng Q4Eng;

typedef struct {
    int max_pos;            // context positions (0: 4096)
    int vram_experts;       // the routed experts in VRAM too (a GPU with room for the pack but the PLE table)
    const char* ple_gguf;   // the PLE n-gram table from this GGUF (Unsloth's engram files), not from the model folder
    double cache_gb;        // > 0: a VRAM cache of host experts (LRU, at least one layer's); < 0: all free VRAM; 0: none
    int stream_experts;     // with cache_gb: misses read from the folder's files, no pinned copy (RAM below the experts)
    int rows;               // rows of one prompt forward (0: Q4E_MAX_ROWS or Q4E_BIG_ROWS), from Q4E_LOGIT_ROWS to the context
    int mtp;                // 1 to Q4E_MAX_DRAFTS: the MTP head (mtp.*) for up to mtp drafts; every forward runs it one row behind
    int tail_skip;          // 1 to top_k - 1, decode with a cache (lossy): a missed expert that every row routes at rank >= tail_skip
                            // is not copied or computed (0: off). CUDA only: Metal holds every expert
    int cpu_tail;           // decode with a cache: up to cpu_tail missed experts a layer (lowest rank first) computed by the CPU
                            // from the pinned arena, beside the GPU's copies (exact; P = 3 seeds; 0: off; at most 16)
    const char* resource_dir; // Metal: the folder of kernels_moe.metallib and kernels_qwen4.metallib (EngOpts.resource_dir,
                            // the harnesses' --res; NULL: out/res). NSLM_RES, when set, overrides it
} Q4EOpts;

Q4Eng* q4e_open(const char* model_dir, const Q4EOpts* o, char* err, int errlen);
void q4e_close(Q4Eng* e);
const Qwen4Cfg* q4e_cfg(const Q4Eng* e);
// Appends ids[0..n-1] to the context; the logits of each row go to logits[n][vocab] (NULL: none).  0, or -1 (q4e_error).
int q4e_forward(Q4Eng* e, const int32_t* ids, int n, float* logits);
void q4e_reset(Q4Eng* e);   // an empty context
// MTP (Q4EOpts.mtp): k <= Q4E_MAX_DRAFTS greedy drafts. Draft 1: the head on the bonus (the last row's arg max, ties to the
// lowest id) and that row's streams; draft i + 1 on draft i and the head's own streams. The caller commits the bonus.
#define Q4E_MAX_DRAFTS 3
int q4e_draft(Q4Eng* e, int k, int32_t* drafts);
// q4e_forward of n <= Q4E_MAX_DRAFTS + 1 rows that keeps the recurrent state (DeltaNet, conv, PLE) after each row;
// q4e_accept(e, m) then keeps rows [0, m) of it: the position and the state of row m - 1. 0, or -1.
int q4e_forward_keep(Q4Eng* e, const int32_t* ids, int n, float* logits);
int q4e_accept(Q4Eng* e, int m);
void q4e_cache_stats(Q4Eng* e, uint64_t* hits, uint64_t* misses, int* slots);   // zeros without a cache
int q4e_pos(const Q4Eng* e);
const char* q4e_error(const Q4Eng* e);
