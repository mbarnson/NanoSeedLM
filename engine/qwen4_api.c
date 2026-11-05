// engine/qwen4_api.c - engine_api.h over the qwen4 CUDA engine, one sequence slot. A prompt that extends what the
// engine holds continues its recurrent state; other prompts and rewinds compute from the start.
#include "engine_qwen4.h"
#include "engine_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen4_eng.h"

struct Eng {   // Q4Api: hist[0 .. n) committed; the engine holds hist[0 .. at), at = n - 1, or n after eng_step
    Q4Eng* q;
    int vocab, max_pos, cap, n, at;
    int32_t* hist;
    float* last;   // eng_step's logits while at == n
    char desc[192];
};

static int fail_reset(Q4Api* e) { q4e_reset(e->q); e->n = e->at = 0; return -1; }
// The engine through hist[at .. to)
static int fwd(Q4Api* e, int to, float* logits) {
    if (to <= e->at) return 0;
    if (q4e_forward(e->q, e->hist + e->at, to - e->at, logits)) return fail_reset(e);
    e->at = to;
    return 0;
}
static int reserve(Q4Api* e, int n) {
    if (n <= e->cap) return 0;
    const int cap = n > 2 * e->cap ? n : 2 * e->cap;
    int32_t* h = (int32_t*) realloc(e->hist, sizeof(int32_t) * (size_t) cap);
    if (!h) return -1;
    e->hist = h;
    e->cap = cap;
    return 0;
}
// The history ids[0 .. n): the engine keeps its state if it holds a prefix of ids, else starts over. *reused: positions kept
static int begin(Q4Api* e, int seq, const int32_t* ids, int n, int* reused) {
    if (seq || n < 1 || n > e->max_pos || reserve(e, n)) return -1;
    if (e->at > n || memcmp(e->hist, ids, sizeof(int32_t) * (size_t) e->at)) { q4e_reset(e->q); e->at = 0; }
    memmove(e->hist, ids, sizeof(int32_t) * (size_t) n);
    e->n = n;
    if (reused) *reused = e->at;
    return 0;
}

static int env_int(const char* k) { return getenv(k) ? atoi(getenv(k)) : 0; }   // nslm-qwen4-gen's options

Eng* eng_open(const EngOpts* o, char* err, int errlen) {
    if (o->max_seqs > 1 || o->kv_format) {
        snprintf(err, (size_t) errlen, "Qwen3.8-Flash-Next: one sequence slot and the f32 KV cache only (max_seqs %d, kv %d)", o->max_seqs, o->kv_format);
        return NULL;
    }
    const char* mb = getenv("NSLM_EXPERT_VRAM_MB");   // else the expert cache takes the free VRAM
    Q4EOpts qo = {(int) (o->kv_tokens > 0 ? o->kv_tokens : 4096), env_int("Q4E_VRAM_EXPERTS"), getenv("Q4E_PLE_GGUF"),
                  mb ? atof(mb) * 1048576.0 / 1e9 : -1, env_int("Q4E_STREAM_EXPERTS"), env_int("Q4E_ROWS"), 0, 0, env_int("Q4E_CPU_TAIL")};   // no MTP head
    Q4Api* e = (Q4Api*) calloc(1, sizeof *e);
    if (!e || !(e->q = q4e_open(o->model_dir, &qo, err, errlen))) { free(e); return NULL; }
    const Qwen4Cfg* c = q4e_cfg(e->q);
    e->vocab = c->vocab;
    e->max_pos = qo.max_pos;
    if (!(e->last = (float*) malloc(sizeof(float) * (size_t) e->vocab))) { eng_close(e); snprintf(err, (size_t) errlen, "out of memory"); return NULL; }
    uint64_t h, m;
    int slots;
    q4e_cache_stats(e->q, &h, &m, &slots);
    snprintf(e->desc, sizeof e->desc, "Qwen3.8-Flash-Next: %d layers, %d experts, %d cache slots, one sequence of %d positions",
             c->n_layer, c->n_exp, slots, e->max_pos);
    return e;
}
void eng_close(Eng* e) {
    if (!e) return;
    q4e_close(e->q);
    free(e->hist); free(e->last); free(e);
}
const char* eng_describe(Eng* e) { return e->desc; }
int eng_vocab(Eng* e) { return e->vocab; }
void eng_mem(Eng* e, EngMem* m) { (void) e; memset(m, 0, sizeof *m); }

int eng_prefill(Eng* e, int seq, const int32_t* ids, int n) {
    q4e_reset(e->q);
    e->at = 0;
    return eng_prefill_cached(e, seq, ids, n, NULL);
}
int eng_prefill_cached(Eng* e, int seq, const int32_t* ids, int n, int* reused) {
    return begin(e, seq, ids, n, reused) ? -1 : fwd(e, n - 1, NULL);
}
int eng_prefill_begin(Eng* e, int seq, const int32_t* ids, int n, int* reused) { return begin(e, seq, ids, n, reused); }
int eng_prefill_next(Eng* e, int seq, int max_rows) {
    if (seq || max_rows < 1) return -1;
    const int left = e->n - 1 - e->at, to = e->at + (max_rows < left ? max_rows : left);   // max_rows may be INT_MAX
    return fwd(e, to, NULL) ? -1 : e->n - 1 - e->at > 0 ? e->n - 1 - e->at : 0;
}
int eng_rewind(Eng* e, int seq, int n) {
    if (seq || n < 1 || n > e->n) return -1;
    if (n < e->n) { q4e_reset(e->q); e->at = 0; e->n = n; }
    return fwd(e, n - 1, NULL);
}
int eng_fork(Eng* e, int dst, int src) { (void) e; return dst == 0 && src == 0 ? 0 : -1; }
void eng_free(Eng* e, int seq) { if (!seq) { q4e_reset(e->q); e->n = e->at = 0; } }
int eng_len(Eng* e, int seq) { return seq ? 0 : e->n; }

int eng_step(Eng* e, int seq, float* logits) {
    if (seq || e->n < 1 || e->at < e->n - 1) return -1;
    if (e->at < e->n && fwd(e, e->n, e->last)) return -1;
    memcpy(logits, e->last, sizeof(float) * (size_t) e->vocab);
    return 0;
}
int eng_push(Eng* e, int seq, int32_t tok) {
    if (seq || e->n < 1 || e->at < e->n - 1 || e->n >= e->max_pos || fwd(e, e->n, NULL) || reserve(e, e->n + 1)) return -1;
    e->hist[e->n++] = tok;
    return 0;
}
int eng_step_batch(Eng* e, const int* seqs, int n, float* logits) { return n == 1 ? eng_step(e, seqs[0], logits) : -1; }
int eng_generate(Eng* e, const int* seqs, int nseq, int n_new, int mode, int32_t* out, EngStats* st) {
    (void) mode;   // every mode commits greedy decode's tokens: here, plain greedy
    if (nseq != 1 || seqs[0]) return -1;
    for (int j = 0; j < n_new; ++j) {
        if (eng_step(e, 0, e->last)) return -1;
        int b = 0;
        for (int v = 1; v < e->vocab; ++v) if (e->last[v] > e->last[b]) b = v;
        if (eng_push(e, 0, b)) return -1;
        out[j] = b;
    }
    if (st) { st->tokens += n_new; st->forwards += n_new; st->rows += n_new; st->cycles += n_new; }
    return 0;
}

int64_t eng_kv_bytes(Eng* e) { (void) e; return 0; }   // no KV files: the recurrent state is not in the KV cache
int eng_kv_read(Eng* e, int seq, int p0, int p1, void* dst) { (void) e; (void) seq; (void) p0; (void) p1; (void) dst; return -1; }
int eng_kv_write(Eng* e, int seq, const int32_t* ids, int p0, int p1, const void* src) {
    (void) e; (void) seq; (void) ids; (void) p0; (void) p1; (void) src;
    return -1;
}

int eng_score(Eng* e, int seq, const int32_t* ids, int from, int count, float* logits) {
    if (seq || from < 1 || count < 1 || begin(e, 0, ids, from + count, NULL)) return -1;
    q4e_reset(e->q);
    e->at = 0;
    return fwd(e, from - 1, NULL) || fwd(e, from + count - 1, logits) ? -1 : 0;
}
