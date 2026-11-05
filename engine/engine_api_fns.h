// engine/engine_api_fns.h - engine_api.h's functions, without an include guard: engine/engine.c includes it once for
// each engine's names (engine_mova.h, engine_qwen4.h). Include engine_api.h, not this file.
struct Eng* eng_open(const EngOpts* o, char* err, int errlen);
void eng_close(struct Eng* e);
const char* eng_describe(struct Eng* e);   // one line: formats, default mode, ...
int eng_vocab(struct Eng* e);
void eng_mem(struct Eng* e, EngMem* m);

// Replace slot `seq` with the prompt ids[0..n-1] (n >= 1) and fill the KV cache for ids[0..n-2].  0 on success.
int eng_prefill(struct Eng* e, int seq, const int32_t* ids, int n);
// As eng_prefill, but keeps the KV cache of the longest common prefix with the slot's history (*reused tokens) and
// computes only the rest.  Prompt rows always take the prefill kernels, whose results do not depend on the chunks: the
// cache and logits are eng_prefill's, bit for bit.
int eng_prefill_cached(struct Eng* e, int seq, const int32_t* ids, int n, int* reused);
// eng_prefill_cached in steps (a server interleaving prompts with decode): eng_prefill_begin sets the history and keeps
// the reusable cache (*reused, may be NULL) but computes nothing; each eng_prefill_next computes up to max_rows more
// positions and returns how many remain (0: ready to step), or -1.  The slot cannot step until 0.
int eng_prefill_begin(struct Eng* e, int seq, const int32_t* ids, int n, int* reused);
int eng_prefill_next(struct Eng* e, int seq, int max_rows);
// Keep the first n committed tokens of `seq` (1 <= n <= length).
int eng_rewind(struct Eng* e, int seq, int n);
// Make slot dst an exact copy of slot src (history and KV).
int eng_fork(struct Eng* e, int dst, int src);
// Release a slot's KV capacity.
void eng_free(struct Eng* e, int seq);
int eng_len(struct Eng* e, int seq);

// Losslessness: every mode must commit exactly the tokens ENG_MODE_AR commits; eng_step's and eng_score's logits are
// the logits greedy decode commits by.
//
// Greedy decode exactly n_new tokens for each listed sequence, appending them to the histories.  out[i * n_new + j]
// is the j-th new token of seqs[i].  Stats are added to *st (may be NULL).
int eng_generate(struct Eng* e, const int* seqs, int nseq, int n_new, int mode, int32_t* out, EngStats* st);

// One decode step without committing: consumes the pending token of `seq` (writes its KV) and returns the
// full-vocabulary logits for the next token; the caller commits a token with eng_push.  Calling eng_step twice
// without eng_push recomputes the same position.
int eng_step(struct Eng* e, int seq, float* logits);
int eng_push(struct Eng* e, int seq, int32_t tok);
// eng_step for n distinct sequences at once (their own positions), in forwards of up to 8 rows: logits[i * vocab ..]
// for seqs[i], as eng_step gives that sequence alone (Metal: bit for bit).
int eng_step_batch(struct Eng* e, const int* seqs, int n, float* logits);

// A slot's KV cache as bytes (saving it to disk): eng_kv_bytes is the size of one position over all layers.
// eng_kv_read copies cached positions [p0, p1) of slot seq to dst; it may run on another thread while the slot is idle.
// eng_kv_write fills positions [p0, p1) (p0 <= the slot's cached positions) from src, with the history ids[0 .. p1):
// the slot then holds p1 tokens, all cached.  Bytes are only meaningful to the same engine, model and KV format.
int64_t eng_kv_bytes(struct Eng* e);
int eng_kv_read(struct Eng* e, int seq, int p0, int p1, void* dst);
int eng_kv_write(struct Eng* e, int seq, const int32_t* ids, int p0, int p1, const void* src);

// Teacher-forced scoring: for j in [0, count), logits[j * vocab ..] are the full-vocabulary logits predicting
// ids[from + j] from ids[0 .. from + j - 1].  Uses slot `seq` as scratch.
int eng_score(struct Eng* e, int seq, const int32_t* ids, int from, int count, float* logits);
