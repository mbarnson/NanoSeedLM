// engine/engine.c - engine_api.h for every model: eng_open takes the qwen4 engine (engine/qwen4_api.c) when config.json
// is a qwen4_exp configuration, else the MoVA engine. engine_mova.h and engine_qwen4.h give each engine its own names.
#include <stdlib.h>

#include "engine_api.h"
#include "qwen4_cfg.h"

struct MovaEng;
struct Q4Api;
#include "engine_mova.h"
#include "engine_api_fns.h"
#include "engine_unname.h"
#include "engine_qwen4.h"
#include "engine_api_fns.h"
#include "engine_unname.h"

struct Eng {
    int qwen;
    void* h;
};
#define CALL(f, ...) (e->qwen ? q4_##f(e->h, __VA_ARGS__) : mova_##f(e->h, __VA_ARGS__))
#define CALL0(f) (e->qwen ? q4_##f(e->h) : mova_##f(e->h))

Eng* eng_open(const EngOpts* o, char* err, int errlen) {
    Qwen4Cfg c;
    Eng* e = (Eng*) calloc(1, sizeof *e);
    e->qwen = o && o->model_dir && !qwen4_cfg_load(&c, o->model_dir, err, errlen);
    err[0] = 0;
    e->h = e->qwen ? (void*) q4_eng_open(o, err, errlen) : (void*) mova_eng_open(o, err, errlen);
    if (!e->h) { free(e); return NULL; }
    return e;
}
void eng_close(Eng* e) {
    if (!e) return;
    CALL0(eng_close);
    free(e);
}
const char* eng_describe(Eng* e) { return CALL0(eng_describe); }
int eng_vocab(Eng* e) { return CALL0(eng_vocab); }
void eng_mem(Eng* e, EngMem* m) { CALL(eng_mem, m); }
int eng_prefill(Eng* e, int seq, const int32_t* ids, int n) { return CALL(eng_prefill, seq, ids, n); }
int eng_prefill_cached(Eng* e, int seq, const int32_t* ids, int n, int* reused) {
    return CALL(eng_prefill_cached, seq, ids, n, reused);
}
int eng_prefill_begin(Eng* e, int seq, const int32_t* ids, int n, int* reused) {
    return CALL(eng_prefill_begin, seq, ids, n, reused);
}
int eng_prefill_next(Eng* e, int seq, int max_rows) { return CALL(eng_prefill_next, seq, max_rows); }
int eng_rewind(Eng* e, int seq, int n) { return CALL(eng_rewind, seq, n); }
int eng_fork(Eng* e, int dst, int src) { return CALL(eng_fork, dst, src); }
void eng_free(Eng* e, int seq) { CALL(eng_free, seq); }
int eng_len(Eng* e, int seq) { return CALL(eng_len, seq); }
int eng_generate(Eng* e, const int* seqs, int nseq, int n_new, int mode, int32_t* out, EngStats* st) {
    return CALL(eng_generate, seqs, nseq, n_new, mode, out, st);
}
int eng_step(Eng* e, int seq, float* logits) { return CALL(eng_step, seq, logits); }
int eng_push(Eng* e, int seq, int32_t tok) { return CALL(eng_push, seq, tok); }
int eng_step_batch(Eng* e, const int* seqs, int n, float* logits) { return CALL(eng_step_batch, seqs, n, logits); }
int64_t eng_kv_bytes(Eng* e) { return CALL0(eng_kv_bytes); }
int eng_kv_read(Eng* e, int seq, int p0, int p1, void* dst) { return CALL(eng_kv_read, seq, p0, p1, dst); }
int eng_kv_write(Eng* e, int seq, const int32_t* ids, int p0, int p1, const void* src) {
    return CALL(eng_kv_write, seq, ids, p0, p1, src);
}
int eng_score(Eng* e, int seq, const int32_t* ids, int from, int count, float* logits) {
    return CALL(eng_score, seq, ids, from, count, logits);
}
