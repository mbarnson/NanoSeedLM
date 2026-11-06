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
    int drafts;    // Q4E_DRAFTS: eng_generate's MTP drafts a forward (0: none, and no MTP head)
    int lrow;      // last holds the logits of hist[lrow] (-1: none), the engine's last row when lrow == at - 1
    int lmax;      // the arg max of last's first row (-1: not computed yet)
    int32_t* hist;
    float* last;   // eng_step's logits while at == n; eng_generate's verify rows (drafts + 1 of them)
    char desc[192];
};

static int fail_reset(Q4Api* e) { q4e_reset(e->q); e->n = e->at = 0; e->lrow = e->lmax = -1; return -1; }
// The engine through hist[at .. to)
static int fwd(Q4Api* e, int to, float* logits) {
    if (to <= e->at) return 0;
    if (q4e_forward(e->q, e->hist + e->at, to - e->at, logits)) return fail_reset(e);
    e->at = to;
    e->lrow = logits == e->last ? to - 1 : -1;
    if (logits == e->last) e->lmax = -1;
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
    const int drafts = env_int("Q4E_DRAFTS");   // the MTP head opens only for drafts: every forward runs it
    if (drafts < 0 || drafts > Q4E_MAX_DRAFTS) {
        snprintf(err, (size_t) errlen, "Q4E_DRAFTS: 0 to %d MTP drafts a forward, not %d", Q4E_MAX_DRAFTS, drafts);
        return NULL;
    }
    const char* mb = getenv("NSLM_EXPERT_VRAM_MB");   // else the expert cache takes the free VRAM
    Q4EOpts qo = {(int) (o->kv_tokens > 0 ? o->kv_tokens : 4096), env_int("Q4E_VRAM_EXPERTS"), getenv("Q4E_PLE_GGUF"),
                  mb ? atof(mb) * 1048576.0 / 1e9 : -1, env_int("Q4E_STREAM_EXPERTS"), env_int("Q4E_ROWS"), drafts, 0, env_int("Q4E_CPU_TAIL")};
    qo.resource_dir = o->resource_dir;   // the Metal engine's metallibs (nslm-serve's ../res beside the binary, --res)
    Q4Api* e = (Q4Api*) calloc(1, sizeof *e);
    if (!e || !(e->q = q4e_open(o->model_dir, &qo, err, errlen))) { free(e); return NULL; }
    const Qwen4Cfg* c = q4e_cfg(e->q);
    e->vocab = c->vocab;
    e->max_pos = qo.max_pos;
    e->drafts = drafts;
    e->lrow = e->lmax = -1;
    if (!(e->last = (float*) malloc(sizeof(float) * (size_t) e->vocab * (size_t) (drafts + 1)))) { eng_close(e); snprintf(err, (size_t) errlen, "out of memory"); return NULL; }
    uint64_t h, m;
    int slots;
    q4e_cache_stats(e->q, &h, &m, &slots);
    snprintf(e->desc, sizeof e->desc, "Qwen3.8-Flash-Next: %d layers, %d experts, %d cache slots, one sequence of %d positions%s",
             c->n_layer, c->n_exp, slots, e->max_pos, drafts == 1 ? ", 1 MTP draft a forward" : drafts == 2 ? ", 2 MTP drafts a forward" :
             drafts == 3 ? ", 3 MTP drafts a forward" : "");
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
static int amax(const float* l, int n) {   // ties: the lowest id
    int b = 0;
    float x = l[0];
    for (int v = 1; v < n; ++v) if (l[v] > x) { x = l[v]; b = v; }
    return b;
}
static int last_max(Q4Api* e) { if (e->lmax < 0) e->lmax = amax(e->last, e->vocab); return e->lmax; }
// Greedy decode. Each token is the bonus (the arg max of the logits so far) or an accepted draft. With Q4E_DRAFTS = K,
// the forward of a pending token that is the bonus of the row before it (an earlier bonus, or eng_push after eng_step)
// carries up to K drafts of the MTP head; a draft is committed while it equals the arg max of the row before it. Those
// are greedy decode's tokens, bit for bit (gate MB): a verify row's logits are a one-row forward's. Other pending tokens
// (a prompt's last, a token not greedy) forward alone. The last bonus waits uncomputed (at == n - 1), as after eng_push,
// so the next call drafts on it.
int eng_generate(Eng* e, const int* seqs, int nseq, int n_new, int mode, int32_t* out, EngStats* st) {
    (void) mode;   // every mode commits greedy decode's tokens: here, greedy decode with Q4E_DRAFTS drafts a forward
    if (nseq != 1 || seqs[0] || n_new < 0) return -1;
    const int V = e->vocab;
    int64_t fw = 0, rows = 0, bonus = 0, prop = 0, acc = 0;
    int j = 0, rc = 0;
    while (j < n_new) {
        if (e->n < 1 || e->at < e->n - 1 || reserve(e, e->n + e->drafts + 1)) { rc = -1; break; }
        if (e->at == e->n) {   // the bonus
            if (e->n >= e->max_pos) { rc = -1; break; }
            out[j++] = e->hist[e->n++] = last_max(e);
            ++bonus;
            continue;
        }
        int k = e->drafts && e->lrow >= 0 && e->lrow == e->n - 2 && last_max(e) == e->hist[e->n - 1] ? e->drafts : 0;
        if (k > n_new - j) k = n_new - j;
        if (k > e->max_pos - e->n) k = e->max_pos - e->n;   // the drafts' positions n .. n + k - 1
        if (!k) {   // the pending token alone
            if (fwd(e, e->n, e->last)) { rc = -1; break; }
            ++fw; ++rows;
            continue;
        }
        int32_t row[Q4E_MAX_DRAFTS + 1];
        int m = 1, am = -1;
        row[0] = e->hist[e->n - 1];
        if (q4e_draft(e->q, k, row + 1) || q4e_forward_keep(e->q, row, k + 1, e->last)) { rc = fail_reset(e); break; }
        while (m <= k && (am = amax(e->last + (size_t) (m - 1) * V, V)) == row[m]) ++m;   // rows 0 .. m - 1 hold
        if (q4e_accept(e->q, m)) { rc = fail_reset(e); break; }
        if (m > 1) memmove(e->last, e->last + (size_t) (m - 1) * V, sizeof(float) * (size_t) V);
        memcpy(e->hist + e->n, row + 1, sizeof(int32_t) * (size_t) (m - 1));
        memcpy(out + j, row + 1, sizeof(int32_t) * (size_t) (m - 1));
        e->n += m - 1;
        j += m - 1;
        e->at = e->n;
        e->lrow = e->n - 1;
        e->lmax = m <= k ? am : -1;   // a rejected draft: row m - 1's arg max is known
        ++fw; rows += k + 1; prop += k; acc += m - 1;
    }
    if (st) { st->tokens += j; st->forwards += fw; st->rows += rows; st->cycles += bonus; st->proposals += prop; st->accepted += acc; }
    return rc;
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
