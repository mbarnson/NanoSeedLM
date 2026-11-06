// harness/sample.c - see sample.h.  The scratch rows are static: each caller samples from one thread.
#include "sample.h"

#include <math.h>
#include <stdlib.h>

#include "rng.h"

uint64_t sample_rand(uint64_t* s) { return ns_splitmix64(s); }
static double unit(Sampler* s) { return ns_unit(&s->rng); }

typedef struct { float p; int id; } Cand;
static int cand_cmp(const void* a, const void* b) {
    const Cand *x = (const Cand*) a, *y = (const Cand*) b;
    return x->p > y->p ? -1 : x->p < y->p ? 1 : x->id - y->id;
}

// The first of n weights (w[i], or c[i].p with id c[i].id) whose running sum exceeds u; zero weights never
static int32_t pick_run(const double* w, const Cand* c, int n, double u) {
    double a = 0;
    int last = -1;
    for (int i = 0; i < n; ++i) {
        const double x = c ? (double) c[i].p : w[i];
        if (x <= 0) continue;
        a += x;
        last = c ? c[i].id : i;
        if (a > u) return last;
    }
    return last;
}
static double q_of(const EngDraftQ* q, int32_t draft, int32_t id) {   // q NULL: one point at the draft
    if (!q) return id == draft;
    for (int i = 0; i < q->n; ++i) if (q->id[i] == id) return q->p[i];
    return 0;
}

int32_t sample_pick(Sampler* s, const float* l, int V, int32_t draft, const EngDraftQ* q) {
    int b = 0;
    for (int i = 1; i < V; ++i) if (l[i] > l[b]) b = i;
    if (s->temp <= 0) return b;
    const double mx = l[b];
    double tot = 0, wd = 0;
    static double* w = NULL;
    static Cand* c = NULL;
    static int cap = 0;
    if (cap < V) { free(w); free(c); w = (double*) malloc(sizeof(double) * (size_t) V); c = (Cand*) malloc(sizeof(Cand) * (size_t) V); cap = V; }
    if (s->mlx) {   // every id, in id order
        for (int i = 0; i < V; ++i) tot += w[i] = exp(((double) l[i] - mx) / s->temp);
        if (draft < 0 || draft >= V) return pick_run(w, NULL, V, unit(s) * tot);
        if (unit(s) * q_of(q, draft, draft) * tot < w[draft]) return draft;   // u < p(d) / q(d)
        double r = tot;   // then w becomes max(0, p - q) tot, of sum r
        for (int i = 0; i < (q ? q->n : 1); ++i) {
            const int32_t id = q ? q->id[i] : draft;
            const double m = q_of(q, draft, id) * tot;
            if (id >= 0 && id < V) { r -= w[id] < m ? w[id] : m; w[id] = w[id] > m ? w[id] - m : 0; }
        }
        return pick_run(w, NULL, V, unit(s) * r);
    }
    const float th = (float) mx - (float) (s->temp * 18.420681);   // below: < 1e-8 each
    int n = 0;
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
    if (draft < 0) return pick_run(NULL, c, keep, unit(s) * tot);
    if (unit(s) * q_of(q, draft, draft) * tot < wd) return draft;   // u < p(d) / q(d)
    double r = 0;   // then c holds max(0, p - q) tot
    for (int i = 0; i < keep; ++i) {
        const double m = q_of(q, draft, c[i].id) * tot;
        c[i].p = c[i].p > m ? (float) (c[i].p - m) : 0;
        r += c[i].p;
    }
    return pick_run(NULL, c, keep, unit(s) * r);
}
int32_t sample_engpick(void* s, const float* l, int V, int32_t draft, const EngDraftQ* q) { return sample_pick((Sampler*) s, l, V, draft, q); }
