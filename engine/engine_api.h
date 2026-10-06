// engine/engine_api.h - the C API between the tools and the MoVA GPU engine.
//
// A sequence slot holds its committed token ids hist[0..n-1].  The KV cache holds positions 0..n-2; the last committed
// token hist[n-1] is "pending" (not yet cached) and the next forward consumes it.  Rolling back = shortening the history.
#pragma once
#include <stdint.h>

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
} EngOpts;

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

Eng* eng_open(const EngOpts* o, char* err, int errlen);
void eng_close(Eng* e);
const char* eng_describe(Eng* e);   // one line: formats, default mode, ...
int eng_vocab(Eng* e);
void eng_mem(Eng* e, EngMem* m);

// Replace slot `seq` with the prompt ids[0..n-1] (n >= 1) and fill the KV cache for ids[0..n-2].  0 on success.
int eng_prefill(Eng* e, int seq, const int32_t* ids, int n);
// As eng_prefill, but keeps the KV cache of the longest common prefix with the slot's history (*reused tokens) and
// computes only the rest.  Short tails take the decode kernels, so logits can differ from eng_prefill's in the last bits.
int eng_prefill_cached(Eng* e, int seq, const int32_t* ids, int n, int* reused);
// Keep the first n committed tokens of `seq` (1 <= n <= length).
int eng_rewind(Eng* e, int seq, int n);
// Make slot dst an exact copy of slot src (history and KV).
int eng_fork(Eng* e, int dst, int src);
// Release a slot's KV capacity.
void eng_free(Eng* e, int seq);
int eng_len(Eng* e, int seq);

// Losslessness: every mode must commit exactly the tokens ENG_MODE_AR commits; eng_step's and eng_score's logits are
// the logits greedy decode commits by.
//
// Greedy decode exactly n_new tokens for each listed sequence, appending them to the histories.  out[i * n_new + j]
// is the j-th new token of seqs[i].  Stats are added to *st (may be NULL).
int eng_generate(Eng* e, const int* seqs, int nseq, int n_new, int mode, int32_t* out, EngStats* st);

// One decode step without committing: consumes the pending token of `seq` (writes its KV) and returns the
// full-vocabulary logits for the next token; the caller commits a token with eng_push.  Calling eng_step twice
// without eng_push recomputes the same position.
int eng_step(Eng* e, int seq, float* logits);
int eng_push(Eng* e, int seq, int32_t tok);

// Teacher-forced scoring: for j in [0, count), logits[j * vocab ..] are the full-vocabulary logits predicting
// ids[from + j] from ids[0 .. from + j - 1].  Uses slot `seq` as scratch.
int eng_score(Eng* e, int seq, const int32_t* ids, int from, int count, float* logits);

#ifdef __cplusplus
}
#endif
