// harness/nslm-mova-smoke.m - open the MoVA engine, prefill pre-tokenized ids, greedy-generate, print ids and timing.
//
//   nslm-mova-smoke --model DIR [--res out/res] --ids 0,250018,... [--n 32]
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "engine_api.h"
#include "mova_ext.h"

static const char* opt(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char** argv) {
    @autoreleasepool {
        EngOpts o;
        memset(&o, 0, sizeof o);
        o.model_dir = opt(argc, argv, "--model", NULL);
        o.resource_dir = opt(argc, argv, "--res", "out/res");
        o.max_seqs = 1;
        o.kv_tokens = 4096;
        if (!o.model_dir || !opt(argc, argv, "--ids", NULL)) { fprintf(stderr, "usage: nslm-mova-smoke --model DIR --ids a,b,c [--n N]\n"); return 2; }
        int32_t ids[4096];
        int n = 0;
        char buf[65536];
        snprintf(buf, sizeof buf, "%s", opt(argc, argv, "--ids", ""));
        for (char* t = strtok(buf, ","); t && n < 4096; t = strtok(NULL, ",")) ids[n++] = atoi(t);
        const int nnew = atoi(opt(argc, argv, "--n", "32"));
        char err[512] = "";
        const double t0 = now_s();
        Eng* e = eng_open(&o, err, sizeof err);
        if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
        EngMem m;
        eng_mem(e, &m);
        printf("%s\nload %.1f s; weights %.2f GB, lut %.3f GB, kv %.2f GB, scratch %.2f GB, gpu allocated %.2f GB\n", eng_describe(e),
               now_s() - t0, m.weights / 1e9, m.lut / 1e9, m.kv / 1e9, m.scratch / 1e9, m.gpu_allocated / 1e9);
        const double t1 = now_s();
        if (eng_prefill(e, 0, ids, n)) { fprintf(stderr, "prefill failed\n"); return 1; }
        const double t2 = now_s();
        int32_t* out = (int32_t*) malloc(sizeof(int32_t) * (size_t) nnew);
        if (eng_generate(e, (int[]){0}, 1, nnew, 0, out, NULL)) { fprintf(stderr, "generate failed\n"); return 1; }
        const double t3 = now_s();
        printf("prefill %d tokens %.2f s (%.1f tok/s); decode %d tokens %.2f s (%.2f tok/s)\nids:", n, t2 - t1, (n - 1) / (t2 - t1),
               nnew, t3 - t2, nnew / (t3 - t2));
        for (int i = 0; i < nnew; ++i) printf("%s%d", i ? "," : " ", out[i]);
        printf("\n");
        eng_close(e);
        return 0;
    }
}
