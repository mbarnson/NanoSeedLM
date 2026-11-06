// tests/test_sample.c - harness/sample.h: the arg max at temperature 0, nslm-chat's and nslm-serve's samplers as they
// were, and the draft rule of speculative sampling: whatever the draft, each token keeps the sampler's distribution.
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

static void freq(Sampler s, int draft, double* f) {
    int n[V] = {0};
    for (int k = 0; k < N; ++k) ++n[sample_pick(&s, kL, V, draft)];
    for (int i = 0; i < V; ++i) f[i] = (double) n[i] / N;
}

int main(void) {
    // Temperature 0: the arg max, ties to the lowest id, whatever the draft
    Sampler g = {0, 1, 0, 0, 0, 7};
    CHECK(sample_pick(&g, kL, V, -1) == 1 && sample_pick(&g, kL, V, 2) == 1 && sample_pick(&g, kL, V, 1) == 1, "arg max");

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
            same &= sample_pick(&s, kL, V, -1) == want;
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
            freq(ss[c], drafts[d], f);
            double worst = 0;
            for (int i = 0; i < V; ++i) worst = fabs(f[i] - p[i]) > worst ? fabs(f[i] - p[i]) : worst;
            CHECK(worst < 0.0015, "sampler %d, draft %d: |freq - p| up to %.4f", c, drafts[d], worst);
        }
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
