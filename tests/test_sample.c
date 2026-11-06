// tests/test_sample.c - harness/sample.h: the arg max at temperature 0, nslm-chat's and nslm-serve's samplers as they
// were, and the draft rule of speculative sampling: whatever the draft and its distribution q, each token keeps the
// sampler's distribution.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../harness/sample.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

enum { V = 6, N = 2000000 };
static const float kL[V] = {1.0f, 2.5f, 2.5f, 0.2f, -1.0f, 1.7f};

// The target distribution of a sampler, by its own definition: p ~ exp((l - max) / T), then the filters
static void target(const Sampler* s, double* p) {
    double mx = kL[0], w[V], tot = 0;
    for (int i = 1; i < V; ++i) mx = kL[i] > mx ? kL[i] : mx;
    for (int i = 0; i < V; ++i) tot += w[i] = exp((kL[i] - mx) / s->temp);
    int ord[V];
    for (int i = 0; i < V; ++i) ord[i] = i;
    for (int i = 0; i < V; ++i)   // by weight, then by id
        for (int j = i + 1; j < V; ++j)
            if (w[ord[j]] > w[ord[i]] || (w[ord[j]] == w[ord[i]] && ord[j] < ord[i])) { const int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    int keep = V;
    double cum = 0;
    for (int i = 0; i < V; ++i) { cum += w[ord[i]] / tot; if (cum >= s->top_p) { keep = i + 1; break; } }
    if (s->top_k > 0 && keep > s->top_k) keep = s->top_k;
    while (keep > 1 && w[ord[keep - 1]] < s->min_p) --keep;
    double ks = 0;
    for (int i = 0; i < V; ++i) p[i] = 0;
    for (int i = 0; i < keep; ++i) ks += w[ord[i]];
    for (int i = 0; i < keep; ++i) p[ord[i]] = w[ord[i]] / ks;
}

// Token frequencies over N picks: draft fixed (q NULL), or each draft drawn from q; *kept: the share that kept it
static void freq(Sampler s, int draft, const EngDraftQ* q, double* f, double* kept) {
    int n[V] = {0}, same = 0;
    uint64_t r = 99;
    for (int k = 0; k < N; ++k) {
        int d = draft;
        if (q) {   // d ~ q
            const double u = (double) (sample_rand(&r) >> 11) * (1.0 / 9007199254740992.0);
            double a = 0;
            for (int i = 0; i < q->n && (d < 0 || a <= u); ++i) { a += q->p[i]; if (a > u) d = q->id[i]; }
            if (d < 0) d = q->id[q->n - 1];
        }
        const int t = sample_pick(&s, kL, V, d, q);
        ++n[t];
        same += t == d;
    }
    for (int i = 0; i < V; ++i) f[i] = (double) n[i] / N;
    if (kept) *kept = (double) same / N;
}
static double worst_gap(const double* f, const double* p) {
    double w = 0;
    for (int i = 0; i < V; ++i) w = fabs(f[i] - p[i]) > w ? fabs(f[i] - p[i]) : w;
    return w;
}

int main(void) {
    // Temperature 0: the arg max, ties to the lowest id, whatever the draft
    Sampler g = {0, 1, 0, 0, 0, 7};
    CHECK(sample_pick(&g, kL, V, -1, NULL) == 1 && sample_pick(&g, kL, V, 2, NULL) == 1 && sample_pick(&g, kL, V, 1, NULL) == 1, "arg max");

    // mlx: nslm-chat's sampler, u = (splitmix64 >> 11) 2^-53 sum, the first id whose cumulative sum exceeds u
    {
        Sampler s = {0.8, 1, 0, 0, 1, 12345};
        uint64_t r = 12345;
        double mx = 2.5, c[V], t = 0;
        for (int i = 0; i < V; ++i) { t += exp((kL[i] - mx) / 0.8); c[i] = t; }
        int same = 1;
        for (int k = 0; k < 1000; ++k) {
            const double u = (double) (sample_rand(&r) >> 11) * (1.0 / 9007199254740992.0) * t;
            int want = 0;
            while (want < V - 1 && !(c[want] > u)) ++want;
            same &= sample_pick(&s, kL, V, -1, NULL) == want;
        }
        CHECK(same, "nslm-chat's sampler");
    }

    // Each sampler, and each draft (none, the likeliest, a filtered-out one, the rarest kept): the target's distribution
    const Sampler ss[4] = {{0.8, 1, 0, 0, 1, 1}, {0.8, 1, 0, 0, 0, 4}, {1.0, 0.95, 0, 20, 0, 2}, {0.7, 0.9, 0.05, 3, 0, 3}};
    for (int c = 0; c < 4; ++c) {
        double p[V], f[V];
        target(&ss[c], p);
        const int drafts[4] = {-1, 1, 4, 5};
        for (int d = 0; d < 4; ++d) {
            freq(ss[c], drafts[d], NULL, f, NULL);
            const double worst = worst_gap(f, p);
            CHECK(worst < 0.0015, "sampler %d, draft %d: |freq - p| up to %.4f", c, drafts[d], worst);
        }
        // Drafts drawn from q: q = p itself (each draft kept), q near p, q with an id that p filters out, q of one point
        EngDraftQ qs[4];
        memset(qs, 0, sizeof qs);
        for (int i = 0; i < V; ++i) if (p[i] > 0) { qs[0].id[qs[0].n] = i; qs[0].p[qs[0].n++] = (float) p[i]; }
        const float qn[V] = {0.10f, 0.30f, 0.25f, 0.15f, 0.05f, 0.15f};
        for (int i = 0; i < V; ++i) { qs[1].id[i] = i; qs[1].p[i] = qn[i]; }
        qs[1].n = V;
        qs[2].n = 2; qs[2].id[0] = 4; qs[2].p[0] = 0.5f; qs[2].id[1] = 1; qs[2].p[1] = 0.5f;
        qs[3].n = 1; qs[3].id[0] = 2; qs[3].p[0] = 1;
        for (int k = 0; k < 4; ++k) {
            double kept;
            freq(ss[c], -1, &qs[k], f, &kept);
            const double worst = worst_gap(f, p);
            CHECK(worst < 0.0015, "sampler %d, q %d: |freq - p| up to %.4f", c, k, worst);
            CHECK(k || kept > 0.999, "sampler %d: q = p keeps %.4f of the drafts", c, kept);
        }
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
