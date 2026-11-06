// engine/qwen4_api.c - engine_api.h over the qwen4 CUDA engine, one sequence slot. A prompt that extends what the
// engine holds continues its recurrent state; other prompts and rewinds compute from the start.
#include "engine_qwen4.h"
#include "engine_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen4_eng.h"
#include "qwen4_ext.h"

struct Eng {   // Q4Api: hist[0 .. n) committed; the engine holds hist[0 .. at), at = n - 1, or n after eng_step
    Q4Eng* q;
    int vocab, max_pos, cap, n, at;
    int drafts;    // Q4E_DRAFTS: eng_generate's MTP drafts a forward (0: none, and no MTP head)
    float min_p;   // Q4E_DRAFT_MIN_P: a forward verifies the drafts before the first of a lower head probability
    int lrow;      // last holds the logits of hist[lrow] (-1: none), the engine's last row when lrow == at - 1
    int next;      // the token picked from last's first row (-1: not picked yet)
    int32_t* hist;
    float* last;   // eng_step's logits while at == n; eng_generate's verify rows (drafts + 1 of them)
    char desc[192];
};

static int fail_reset(Q4Api* e) { q4e_reset(e->q); e->n = e->at = 0; e->lrow = e->next = -1; return -1; }
// The engine through hist[at .. to)
static int fwd(Q4Api* e, int to, float* logits) {
    if (to <= e->at) return 0;
    if (q4e_forward(e->q, e->hist + e->at, to - e->at, logits)) return fail_reset(e);
    e->at = to;
    e->lrow = logits == e->last ? to - 1 : -1;
    if (logits == e->last) e->next = -1;
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
                  mb ? atof(mb) * 1048576.0 / 1e9 : -1, env_int("Q4E_STREAM_EXPERTS"), env_int("Q4E_ROWS"), drafts, 0, env_int("Q4E_CPU_TAIL"),
                  o->resource_dir, env_int("Q4E_PASS_ROWS")};   // metallibs: nslm-serve's ../res beside the binary, --res
    Q4Api* e = (Q4Api*) calloc(1, sizeof *e);
    if (!e || !(e->q = q4e_open(o->model_dir, &qo, err, errlen))) { free(e); return NULL; }
    const Qwen4Cfg* c = q4e_cfg(e->q);
    e->vocab = c->vocab;
    e->max_pos = qo.max_pos;
    e->drafts = drafts;
    e->min_p = getenv("Q4E_DRAFT_MIN_P") ? (float) atof(getenv("Q4E_DRAFT_MIN_P")) : 0;
    e->lrow = e->next = -1;
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
static int32_t pick_max(void* ctx, const float* l, int V, int32_t draft, const EngDraftQ* q) { (void) ctx; (void) draft; (void) q; return amax(l, V); }
// Each token is the bonus (picked from the logits so far) or an accepted draft. With Q4E_DRAFTS = K, the forward of a
// pending token whose row before it has logits carries up to K drafts of the MTP head, drafted on that token; a draft is
// committed while pick keeps it, and pick's other token takes the first rejected one's place. So each token has pick's
// distribution: with pick_max, greedy decode's tokens, bit for bit (gate MB). Other pending tokens (a prompt's last)
// forward alone. The last bonus waits uncomputed (at == n - 1), as after eng_push, so the next call drafts on it.
int eng_generate_pick(Eng* e, int seq, int n_new, EngPick pick, void* ctx, int32_t* out, EngStats* st) {
    if (seq || n_new < 0 || !pick || e->drafts > ENG_MAX_DRAFTS) return -1;
    const int V = e->vocab;
    int64_t fw = 0, rows = 0, bonus = 0, prop = 0, acc = 0;
    int j = 0, rc = 0;
    while (j < n_new) {
        if (e->n < 1 || e->at < e->n - 1 || reserve(e, e->n + e->drafts + 1)) { rc = -1; break; }
        if (e->at == e->n) {   // the bonus
            if (e->n >= e->max_pos) { rc = -1; break; }
            const int32_t t = e->next >= 0 ? e->next : pick(ctx, e->last, V, -1, NULL);
            e->next = -1;
            if (t < 0) break;   // a stop
            out[j++] = e->hist[e->n++] = t;
            ++bonus;
            continue;
        }
        int k = e->drafts && e->lrow >= 0 && e->lrow == e->n - 2 ? e->drafts : 0;
        if (k > n_new - j) k = n_new - j;
        if (k > e->max_pos - e->n) k = e->max_pos - e->n;   // the drafts' positions n .. n + k - 1
        int32_t row[Q4E_MAX_DRAFTS + 1];
        float pr[Q4E_MAX_DRAFTS];
        row[0] = e->hist[e->n - 1];
        if (k && q4e_draft_from(e->q, row[0], k, row + 1)) { rc = fail_reset(e); break; }
        if (k && e->min_p > 0 && !q4e_draft_probs(e->q, k, pr))   // the gate: the drafts before the first unsure one
            for (int g = 0; g < k; ++g) if (pr[g] < e->min_p) { k = g; break; }
        if (!k) {   // the pending token alone
            if (fwd(e, e->n, e->last)) { rc = -1; break; }
            ++fw; ++rows;
            continue;
        }
        int m = 1, t = -1;
        if (q4e_forward_keep(e->q, row, k + 1, e->last)) { rc = fail_reset(e); break; }
        while (m <= k && (t = pick(ctx, e->last + (size_t) (m - 1) * V, V, row[m], NULL)) == row[m]) ++m;   // rows 0 .. m - 1 hold
        if (q4e_accept(e->q, m)) { rc = fail_reset(e); break; }
        if (m > 1) memmove(e->last, e->last + (size_t) (m - 1) * V, sizeof(float) * (size_t) V);
        memcpy(e->hist + e->n, row + 1, sizeof(int32_t) * (size_t) (m - 1));
        memcpy(out + j, row + 1, sizeof(int32_t) * (size_t) (m - 1));
        e->n += m - 1;
        j += m - 1;
        e->at = e->n;
        e->lrow = e->n - 1;
        e->next = m <= k ? t : -1;   // a rejected draft: row m - 1's pick takes its place
        ++fw; rows += k + 1; prop += k; acc += m - 1;
        if (m <= k && t < 0) break;   // a stop in place of draft m
    }
    if (st) { st->tokens += j; st->forwards += fw; st->rows += rows; st->cycles += bonus; st->proposals += prop; st->accepted += acc; }
    return rc ? -1 : j;
}
int eng_generate(Eng* e, const int* seqs, int nseq, int n_new, int mode, int32_t* out, EngStats* st) {
    (void) mode;   // every mode commits greedy decode's tokens: here, greedy decode with Q4E_DRAFTS drafts a forward
    return nseq != 1 || eng_generate_pick(e, seqs[0], n_new, pick_max, NULL, out, st) != n_new ? -1 : 0;
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
