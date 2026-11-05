// engine/qwen4_eng.h - the qwen4 (qwen4_exp) CUDA engine: an MLX model folder on the GPU, tokens in, logits out.
// Routed experts in mapped host memory (or VRAM), the rest in VRAM; prompts run in rows of up to Q4E_MAX_ROWS tokens.
#pragma once
#include <stdint.h>

#define Q4E_MAX_ROWS 1024    // rows of one forward without logits: fewer forwards copy each missing expert fewer times
#define Q4E_LOGIT_ROWS 128   // rows of one forward with logits (the logits take vocab floats a row)
#define Q4E_PASS_ROWS 32768  // most prompt rows of one layer-major pass (the context, rounded up to chunks): each layer's
                             // missing experts cross PCIe once a pass

#include "qwen4_cfg.h"

typedef struct Q4Eng Q4Eng;

typedef struct {
    int max_pos;            // context positions (0: 4096)
    int vram_experts;       // the routed experts in VRAM too (a GPU with room for the pack but the PLE table)
    const char* ple_gguf;   // the PLE n-gram table from this GGUF (Unsloth's engram files), not from the model folder
    double cache_gb;        // > 0: a VRAM cache of host experts (LRU, at least one layer's experts); 0: read host memory
    int stream_experts;     // with cache_gb: misses read from the folder's files, no pinned copy (RAM below the experts)
} Q4EOpts;

Q4Eng* q4e_open(const char* model_dir, const Q4EOpts* o, char* err, int errlen);
void q4e_close(Q4Eng* e);
const Qwen4Cfg* q4e_cfg(const Q4Eng* e);
// Appends ids[0..n-1] to the context; the logits of each row go to logits[n][vocab] (NULL: none).  0, or -1 (q4e_error).
int q4e_forward(Q4Eng* e, const int32_t* ids, int n, float* logits);
void q4e_reset(Q4Eng* e);   // an empty context
void q4e_cache_stats(Q4Eng* e, uint64_t* hits, uint64_t* misses, int* slots);   // zeros without a cache
int q4e_pos(const Q4Eng* e);
const char* q4e_error(const Q4Eng* e);
