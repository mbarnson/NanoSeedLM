// harness/sample.c - see sample.h.  The scratch rows are static: each caller samples from one thread.
#include "sample.h"

#include <math.h>
#include <stdlib.h>

uint64_t sample_rand(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double unit(Sampler* s) { return (double) (sample_rand(&s->rng) >> 11) * (1.0 / 9007199254740992.0); }

typedef struct { float p; int id; } Cand;
static int cand_cmp(const void* a, const void* b) {
    const Cand *x = (const Cand*) a, *y = (const Cand*) b;
    return x->p > y->p ? -1 : x->p < y->p ? 1 : x->id - y->id;
}

// The first of n weights (skipping id skip) whose running sum exceeds u, as an id; ids[i] or i when ids is NULL
static int32_t pick_run(const double* w, const Cand* c, int n, int skip, double u) {
    double a = 0;
    int last = -1;
    for (int i = 0; i < n; ++i) {
        const int id = c ? c[i].id : i;
        if (id == skip) continue;
        a += c ? (double) c[i].p : w[i];
        last = id;
        if (a > u) return id;
    }
    return last;
}

int32_t sample_pick(Sampler* s, const float* l, int V, int32_t draft) {
    int b = 0;
    for (int i = 1; i < V; ++i) if (l[i] > l[b]) b = i;
    if (s->temp <= 0) return b;
    const double mx = l[b];
    double tot = 0, wd = 0;
    int n;
    static double* w = NULL;
    static Cand* c = NULL;
    static int cap = 0;
    if (cap < V) { free(w); free(c); w = (double*) malloc(sizeof(double) * (size_t) V); c = (Cand*) malloc(sizeof(Cand) * (size_t) V); cap = V; }
    if (s->mlx) {   // every id, in id order
        for (int i = 0; i < V; ++i) tot += w[i] = exp(((double) l[i] - mx) / s->temp);
        n = V;
        if (draft >= 0 && draft < V) wd = w[draft];
        if (draft < 0 || unit(s) * tot >= wd) return pick_run(w, NULL, n, draft, unit(s) * (tot - wd));
        return draft;
    }
    const float th = (float) mx - (float) (s->temp * 18.420681);   // below: < 1e-8 each
    n = 0;
    double all = 0;
    for (int i = 0; i < V; ++i)
        if (l[i] >= th) { c[n].p = (float) exp(((double) l[i] - mx) / s->temp); c[n].id = i; all += c[n].p; ++n; }
    qsort(c, (size_t) n, sizeof(Cand), cand_cmp);
    int keep = n;
    double cum = 0;
    for (int i = 0; i < n; ++i) { cum += c[i].p / all; if (cum >= s->top_p) { keep = i + 1; break; } }
    if (s->top_k > 0 && keep > s->top_k) keep = s->top_k;
    while (keep > 1 && c[keep - 1].p < s->min_p) --keep;   // c[i].p = p_i / p_max
    for (int i = 0; i < keep; ++i) { tot += c[i].p; if (c[i].id == draft) wd = c[i].p; }
    if (draft < 0 || unit(s) * tot >= wd) return pick_run(NULL, c, keep, draft, unit(s) * (tot - wd));
    return draft;
}
