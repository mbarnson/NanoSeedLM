// harness/nslm-mova-plcheck.c - prompt-lookup decoding (ENG_MODE_PL) against plain greedy AR on a prompt suite: the
// committed tokens must be identical (the engine API's losslessness contract); prints acceptance and both speeds.
//
//   nslm-mova-plcheck --model DIR [--res out/res] --prompts PRE [--n 256] [--only id,id]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_api.h"
#include "platform.h"
#include "suite.h"

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    const char* pre = opt(argc, argv, "--prompts", NULL);
    const int n = atoi(opt(argc, argv, "--n", "256"));
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = opt(argc, argv, "--model", NULL);
    o.resource_dir = opt(argc, argv, "--res", "out/res");
    o.max_seqs = 1;
    o.kv_tokens = 20480;
    Suite su;
    if (!o.model_dir || !pre || suite_load(&su, pre)) { fprintf(stderr, "usage: nslm-mova-plcheck --model DIR --prompts PRE [--n 256]\n"); return 2; }
    const char* only = opt(argc, argv, "--only", NULL);
    char err[512] = "";
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
    printf("%s\n", eng_describe(e));
    int32_t* a = (int32_t*) malloc(4 * (size_t) n), *b = (int32_t*) malloc(4 * (size_t) n);
    int prompts = 0, diff = 0;
    double ta = 0, tb = 0;
    long long prop = 0, acc = 0, fw = 0;
    const int seq = 0;
    for (int pi = 0; pi < su.n; ++pi) {
        const SuitePrompt* q = &su.p[pi];
        if (!suite_selected(only, q->id)) continue;
        if (eng_prefill(e, 0, q->ids, q->n)) return 1;
        double t0 = now_s();
        if (eng_generate(e, &seq, 1, n, ENG_MODE_AR, a, NULL)) return 1;
        const double da = now_s() - t0;
        if (eng_prefill(e, 0, q->ids, q->n)) return 1;
        EngStats st;
        memset(&st, 0, sizeof st);
        t0 = now_s();
        if (eng_generate(e, &seq, 1, n, ENG_MODE_PL, b, &st)) return 1;
        const double db = now_s() - t0;
        int k = 0;
        while (k < n && a[k] == b[k]) ++k;
        diff += k < n;
        ++prompts;
        ta += da; tb += db; prop += st.proposals; acc += st.accepted; fw += st.forwards;
        printf("%-8s %s  AR %5.1f tok/s  PL %5.1f tok/s (x%.2f)  accepted %4lld / %4lld  passes %4lld", q->id,
               k < n ? "DIFFER" : "same  ", n / da, n / db, da / db, (long long) st.accepted, (long long) st.proposals, (long long) st.forwards);
        if (k < n) printf("  first difference at token %d", k);
        printf("\n");
        fflush(stdout);
    }
    printf("plcheck: %d prompts, %d differ; AR %.1f tok/s, PL %.1f tok/s (x%.2f); accepted %lld of %lld proposals; %.2f tokens per pass\n",
           prompts, diff, prompts * n / ta, prompts * n / tb, ta / tb, acc, prop, (double) prompts * n / (double) fw);
    free(a); free(b);
    suite_free(&su);
    eng_close(e);
    return diff != 0;
}
