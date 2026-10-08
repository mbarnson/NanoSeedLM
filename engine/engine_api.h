// engine/engine_api.h - the C API between the tools and the MoVA GPU engine.
//
// A sequence slot holds its committed token ids hist[0..n-1].  The KV cache holds positions 0..n-2; the last committed
// token hist[n-1] is "pending" (not yet cached) and the next forward consumes it.  Rolling back = shortening the history.
// Each slot has kv_tokens / max_seqs positions of the KV cache.
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
    int kv_format;             // ENG_KV_BF16 (0, the default: exact), or ENG_KV_Q8 (int8 with a scale per token and
                               // head: half the memory, for long contexts); an engine may only support BF16
                               // (eng_open then fails with "... not supported ...")
    int mla_expand_min;        // MLA (Metal), B (0: 256): a prefill of at least B new tokens attends with the latent
                               // expanded per head (long prompts: faster) up to the last multiple of B, the rest and
                               // shorter prefills (chat turns over a long cached context) in latent space (no expanding
                               // every cached position).  A prompt whose cache is restored in blocks of B and its tail
                               // recomputed gets the cache computed whole, bit for bit.
} EngOpts;
enum { ENG_KV_BF16 = 0, ENG_KV_Q8 = 1 };

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
// computes only the rest.  Prompt rows always take the prefill kernels, whose results do not depend on the chunks: the
// cache and logits are eng_prefill's, bit for bit.
int eng_prefill_cached(Eng* e, int seq, const int32_t* ids, int n, int* reused);
// eng_prefill_cached in steps (a server interleaving prompts with decode): eng_prefill_begin sets the history and keeps
// the reusable cache (*reused, may be NULL) but computes nothing; each eng_prefill_next computes up to max_rows more
// positions and returns how many remain (0: ready to step), or -1.  The slot cannot step until 0.
int eng_prefill_begin(Eng* e, int seq, const int32_t* ids, int n, int* reused);
int eng_prefill_next(Eng* e, int seq, int max_rows);
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
// eng_step for n distinct sequences at once (their own positions), in forwards of up to 8 rows: logits[i * vocab ..]
// for seqs[i], as eng_step gives that sequence alone (Metal: bit for bit).
int eng_step_batch(Eng* e, const int* seqs, int n, float* logits);

// A slot's KV cache as bytes (saving it to disk): eng_kv_bytes is the size of one position over all layers.
// eng_kv_read copies cached positions [p0, p1) of slot seq to dst; it may run on another thread while the slot is idle.
// eng_kv_write fills positions [p0, p1) (p0 <= the slot's cached positions) from src, with the history ids[0 .. p1):
// the slot then holds p1 tokens, all cached.  Bytes are only meaningful to the same engine, model and KV format.
int64_t eng_kv_bytes(Eng* e);
int eng_kv_read(Eng* e, int seq, int p0, int p1, void* dst);
int eng_kv_write(Eng* e, int seq, const int32_t* ids, int p0, int p1, const void* src);

// Teacher-forced scoring: for j in [0, count), logits[j * vocab ..] are the full-vocabulary logits predicting
// ids[from + j] from ids[0 .. from + j - 1].  Uses slot `seq` as scratch.
int eng_score(Eng* e, int seq, const int32_t* ids, int from, int count, float* logits);

#ifdef __cplusplus
}
#endif
