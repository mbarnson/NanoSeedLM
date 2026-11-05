// engine/engine_api.h - the C API between the tools and the MoVA GPU engine.
//
// A sequence slot holds its committed token ids hist[0..n-1].  The KV cache holds positions 0..n-2; the last committed
// token hist[n-1] is "pending" (not yet cached) and the next forward consumes it.  Rolling back = shortening the history.
// Each slot has kv_tokens / max_seqs positions of the KV cache.
#pragma once
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ENG_MODE_DEFAULT = 0,   // AR
    ENG_MODE_AR = 1,        // plain autoregressive greedy decode
    ENG_MODE_PL = 2,        // prompt-lookup speculative decode (lossless)
};

typedef struct {
    const char* model_dir;     // a NanoSeedLM model folder, or the original HF snapshot (BF16)
    const char* resource_dir;  // directory holding kernels_moe.metallib
    int max_seqs;              // sequence slots, ids 0..max_seqs-1
    int64_t kv_tokens;         // total KV capacity across all slots, in tokens
    int kv_format;             // ENG_KV_BF16 (0, the default: exact), or ENG_KV_Q8 (GQA: int8 with a scale per token and
                               // head: half the memory, for long contexts), ENG_KV_FP8 / ENG_KV_FP4 (MLA, Metal: DeepSeek
                               // V4.1's E4M3 with a power-of-two scale per 32 values / E2M1 with an E4M3 scale per 16,
                               // nslm/kvq.h: 52% / 28% of the memory); an engine may support fewer (eng_open then fails
                               // naming the formats it supports)
    int mla_expand_min;        // MLA (Metal), B (0: 256): a prefill of at least B new tokens attends with the latent
                               // expanded per head (long prompts: faster) up to the last multiple of B, the rest and
                               // shorter prefills (chat turns over a long cached context) in latent space (no expanding
                               // every cached position).  A prompt whose cache is restored in blocks of B and its tail
                               // recomputed gets the cache computed whole, bit for bit.
} EngOpts;
enum { ENG_KV_BF16 = 0, ENG_KV_Q8 = 1, ENG_KV_FP8 = 2, ENG_KV_FP4 = 3 };
// A --kv option's value (bf16, q8, fp8, fp4) as ENG_KV_*; -1 for anything else
static inline int eng_kv_parse(const char* s) {
    static const char* const names[4] = {"bf16", "q8", "fp8", "fp4"};
    for (int i = 0; i < 4; ++i)
        if (s && !strcmp(s, names[i])) return i;
    return -1;
}

typedef struct {
    int64_t tokens;      // tokens committed
    int64_t forwards;    // forward passes (a pass may carry many rows)
    int64_t rows;        // rows processed over all passes
    int64_t cycles;      // prompt-lookup verify cycles / AR bursts
    int64_t proposals;   // prompt-lookup proposals offered
    int64_t accepted;    // prompt-lookup proposals accepted
} EngStats;

// Memory the engine holds, by kind (bytes).  weights + lut is the resident weight footprint.
typedef struct {
    int64_t weights;        // model tensors as the kernels read them
    int64_t lut;            // LFSR stream tables
    int64_t kv;             // KV cache arena
    int64_t scratch;        // activations, logits, attention partials
    int64_t staging;        // load-time buffers still alive
    int64_t gpu_allocated;  // MTLDevice.currentAllocatedSize at the call
} EngMem;

typedef struct Eng Eng;

#include "engine_api_fns.h"

#ifdef __cplusplus
}
#endif
