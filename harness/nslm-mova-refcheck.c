// harness/nslm-mova-refcheck.c - the GPU engine against the CPU reference forward (nslm/mova_ref.h) on a real model.
//
//   nslm-mova-refcheck --model DIR [--res out/res] [--threads 16] [--tokens 24] "text"
//
// Encodes the text (with BOS), scores it in both (eng_score and mova_ref_forward), and prints per row the relative
// logit error, whether the arg max agrees, whether every router choice agrees, and the KL divergence of the engine's
// softmax from the reference's; then the totals.  The reference is slow: a few seconds per token on a desktop CPU.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_api.h"
#include "mova_cfg.h"
#include "mova_ext.h"
#include "mova_ref.h"
#include "platform.h"
#include "tokenizer.h"

static double kl(const float* p_ref, const float* q, int V) {   // KL(ref || engine) of the softmaxes, nats
    double mp = -INFINITY, mq = -INFINITY;
    for (int i = 0; i < V; ++i) { mp = fmax(mp, p_ref[i]); mq = fmax(mq, q[i]); }
    double zp = 0, zq = 0;
    for (int i = 0; i < V; ++i) { zp += exp(p_ref[i] - mp); zq += exp(q[i] - mq); }
    double k = 0;
    for (int i = 0; i < V; ++i) {
        const double lp = p_ref[i] - mp - log(zp), lq = q[i] - mq - log(zq);
        k += exp(lp) * (lp - lq);
    }
    return k;
}
static int argmax(const float* l, int V) {
    int b = 0;
    for (int i = 1; i < V; ++i) if (l[i] > l[b]) b = i;
    return b;
}
static int same_set(const int32_t* a, const int32_t* b, int k) {
    for (int i = 0; i < k; ++i) {
        int f = 0;
        for (int j = 0; j < k; ++j) f |= a[i] == b[j];
        if (!f) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    const char* model = opt(argc, argv, "--model", NULL);
    const char* res = opt(argc, argv, "--res", "out/res");
    const int threads = atoi(opt(argc, argv, "--threads", "16")), maxt = atoi(opt(argc, argv, "--tokens", "24"));
    const char* text = argc > 1 ? argv[argc - 1] : NULL;
    if (!model || !text || text[0] == '-') {
        fprintf(stderr, "usage: nslm-mova-refcheck --model DIR [--res out/res] [--threads 16] [--tokens 24] \"text\"\n");
        return 2;
    }
    char err[512] = "", path[2048];
    snprintf(path, sizeof path, "%s/tokenizer.json", model);
    Tok* tok = tok_open(path, err, sizeof err);
    if (!tok) { fprintf(stderr, "tokenizer: %s\n", err); return 1; }
    int32_t ids[4096];
    int n = tok_encode(tok, text, 1, ids, 4096);
    if (n < 2) { fprintf(stderr, "text too short\n"); return 1; }
    if (n > maxt) n = maxt;
    MovaCfg c;
    if (mova_cfg_load(&c, model, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    const int ns = c.n_layer - c.first_sparse, V = c.vocab;

    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = model;
    o.resource_dir = res;
    o.max_seqs = 1;
    o.kv_tokens = n + 64;
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
    fprintf(stderr, "%s\n", eng_describe(e));
    float* el = (float*) malloc(sizeof(float) * (size_t) n * V);
    int32_t* em = (int32_t*) malloc(sizeof(int32_t) * (size_t) n * ns * c.top_k);
    int32_t* ev = (int32_t*) malloc(sizeof(int32_t) * (size_t) n * ns * c.top_kv);
    // --noise-floor: in place of the engine, the reference with exact attention, so the totals show how far two equally
    // valid BF16 forwards (rounding attention at different points) drift apart on this text
    const int floor_only = opt_flag(argc, argv, "--noise-floor");
    MovaRef* r = NULL;
    if (floor_only) {
        eng_close(e);
        r = mova_ref_open(model, threads, err, sizeof err);
        if (!r) { fprintf(stderr, "reference: %s\n", err); return 1; }
        mova_ref_attn_rounding(r, 0);
        if (mova_ref_forward(r, ids, n - 1, 0, el, em, ev)) { fprintf(stderr, "reference failed\n"); return 1; }
    } else {
        eng_mova_routes(e, 1, n);
        if (eng_score(e, 0, ids, 1, n - 1, el) || eng_mova_routes_read(e, n - 1, em, ev, NULL, NULL)) { fprintf(stderr, "engine failed\n"); return 1; }
        eng_close(e);
        r = mova_ref_open(model, threads, err, sizeof err);
        if (!r) { fprintf(stderr, "reference: %s\n", err); return 1; }
    }
    float* rl = (float*) malloc(sizeof(float) * (size_t) n * V);
    int32_t* rm = (int32_t*) malloc(sizeof(int32_t) * (size_t) n * ns * c.top_k);
    int32_t* rv = (int32_t*) malloc(sizeof(int32_t) * (size_t) n * ns * c.top_kv);
    mova_ref_attn_rounding(r, n - 1 > 8 && !opt_flag(argc, argv, "--exact-attn"));   // as the engine's scoring forward
    const double t0 = now_s();
    if (mova_ref_forward(r, ids, n - 1, 0, rl, rm, rv)) { fprintf(stderr, "reference failed\n"); return 1; }
    fprintf(stderr, "reference: %d tokens in %.1f s\n", n - 1, now_s() - t0);

    int am = 0, routes = 0;
    double klsum = 0, klmax = 0, errmax = 0;
    printf("row  token   rel_err   kl        argmax  routes\n");
    for (int t = 0; t < n - 1; ++t) {
        const float* a = el + (size_t) t * V, *b = rl + (size_t) t * V;
        double m = 0, s = 0;
        for (int i = 0; i < V; ++i) { m = fmax(m, fabs((double) a[i] - b[i])); s = fmax(s, fabs((double) b[i])); }
        int same = 1, first = -1, nd = 0;
        for (int l = 0; l < ns; ++l) {
            const int sl = same_set(em + ((size_t) t * ns + l) * c.top_k, rm + ((size_t) t * ns + l) * c.top_k, c.top_k) &&
                           same_set(ev + ((size_t) t * ns + l) * c.top_kv, rv + ((size_t) t * ns + l) * c.top_kv, c.top_kv);
            if (!sl && first < 0) first = c.first_sparse + l;
            nd += !sl;
            same &= sl;
        }
        const double k = kl(b, a, V);
        const int ok = argmax(a, V) == argmax(b, V);
        printf("%3d %7d   %.2e  %.2e  %s     %s", t, ids[t], m / s, k, ok ? "same" : "DIFF", same ? "same" : "differ");
        if (!same) printf(" (%d of %d layers, first %d)", nd, ns, first);
        printf("\n");
        am += ok;
        routes += same;
        klsum += k;
        klmax = fmax(klmax, k);
        errmax = fmax(errmax, m / s);
    }
    printf("rows %d: argmax equal %d, every router choice equal %d, KL mean %.2e max %.2e, relative logit error max %.2e\n",
           n - 1, am, routes, klsum / (n - 1), klmax, errmax);
    mova_ref_close(r);
    tok_close(tok);
    return 0;
}
