// harness/nslm-qwen4-kld.c - KL divergence of a qwen4 model against a reference on one text, through the CUDA engine.
//   nslm-qwen4-kld --ref DIR --model DIR --text FILE [--tokens 1024] [--expert-cache-gb N] [--vram-experts]
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"
#include "qwen4_eng.h"
#include "tokenizer.h"

static int cmp_f(const void* a, const void* b) { const float x = *(const float*) a, y = *(const float*) b; return x < y ? -1 : x > y; }

// Log-probabilities of rows [0, n) of ids under the model at dir into lp[n][vocab]; 0 or -1
static int run(const char* dir, const Q4EOpts* o, const int32_t* ids, int n, float* lp, int vocab) {
    char err[512];
    const double t0 = now_s();
    Q4Eng* e = q4e_open(dir, o, err, sizeof err);
    if (!e) { fprintf(stderr, "%s: %s\n", dir, err); return -1; }
    if (q4e_cfg(e)->vocab != vocab) { fprintf(stderr, "%s: vocab %d, not %d\n", dir, q4e_cfg(e)->vocab, vocab); q4e_close(e); return -1; }
    const double t1 = now_s();
    const int rc = q4e_forward(e, ids, n, lp);
    if (rc) fprintf(stderr, "%s: %s\n", dir, q4e_error(e));
    fprintf(stderr, "%s: loaded in %.1f s, %d rows in %.1f s\n", dir, t1 - t0, n, now_s() - t1);
    q4e_close(e);
    for (int t = 0; !rc && t < n; ++t) {   // log-softmax in double
        float* l = lp + (size_t) t * vocab;
        double mx = l[0], s = 0;
        for (int v = 1; v < vocab; ++v) if (l[v] > mx) mx = l[v];
        for (int v = 0; v < vocab; ++v) s += exp((double) l[v] - mx);
        const double z = mx + log(s);
        for (int v = 0; v < vocab; ++v) l[v] = (float) ((double) l[v] - z);
    }
    return rc;
}

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    const char *ref = NULL, *model = NULL, *text = NULL;
    int ntok = 1024;
    Q4EOpts o = {0, 0, NULL, 0};
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) text = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) ntok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--expert-cache-gb") && i + 1 < argc) o.cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--vram-experts")) o.vram_experts = 1;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!ref || !model || !text || ntok < 2) { fprintf(stderr, "usage: nslm-qwen4-kld --ref DIR --model DIR --text FILE [--tokens N]\n"); return 2; }
    char err[512], path[2048];
    snprintf(path, sizeof path, "%s/tokenizer.json", ref);
    Tok* tok = tok_open(path, err, sizeof err);
    size_t len = 0;
    char* s = plat_slurp(text, &len);
    if (!tok || !s) { fprintf(stderr, "%s\n", tok ? "cannot read the text" : err); return 1; }
    if (len > (size_t) ntok * 16) s[len = (size_t) ntok * 16] = 0;   // enough text for ntok tokens
    int32_t* ids = (int32_t*) malloc(sizeof(int32_t) * (len + 16));
    int n = tok_encode(tok, s, 0, ids, (int) len + 16);
    if (n > ntok) n = ntok;
    o.max_pos = n;
    Qwen4Cfg c;
    if (n < 2 || qwen4_cfg_load(&c, ref, err, sizeof err)) { fprintf(stderr, "%s\n", n < 2 ? "text too short" : err); return 1; }
    const size_t V = (size_t) c.vocab;
    float* lr = (float*) malloc(sizeof(float) * (size_t) n * V);
    float* lm = (float*) malloc(sizeof(float) * (size_t) n * V);
    if (!lr || !lm || run(ref, &o, ids, n, lr, c.vocab) || run(model, &o, ids, n, lm, c.vocab)) return 1;
    float* kl = (float*) malloc(sizeof(float) * (size_t) n);
    double skl = 0, nll_r = 0, nll_m = 0;
    int agree = 0;
    for (int t = 0; t < n; ++t) {
        const float* a = lr + (size_t) t * V;
        const float* b = lm + (size_t) t * V;
        double k = 0;
        size_t ia = 0, ib = 0;
        for (size_t v = 0; v < V; ++v) {
            k += exp((double) a[v]) * ((double) a[v] - b[v]);
            if (a[v] > a[ia]) ia = v;
            if (b[v] > b[ib]) ib = v;
        }
        kl[t] = (float) k;
        skl += k;
        agree += ia == ib;
        if (t + 1 < n) { nll_r -= a[ids[t + 1]]; nll_m -= b[ids[t + 1]]; }
    }
    qsort(kl, (size_t) n, sizeof(float), cmp_f);
    printf("%d tokens: KL(ref || model) mean %.5f, median %.5f, 99th percentile %.5f; top-1 agreement %.2f%%; "
           "perplexity ref %.3f, model %.3f\n", n, skl / n, kl[n / 2], kl[(int) (0.99 * (n - 1))], 100.0 * agree / n,
           exp(nll_r / (n - 1)), exp(nll_m / (n - 1)));
    free(lr); free(lm); free(kl); free(ids); free(s);
    tok_close(tok);
    return 0;
}
