// harness/nslm-mova-plcheck.m - prompt-lookup decoding (ENG_MODE_PL) against plain greedy AR on a prompt suite: the
// committed tokens must be identical (the engine API's losslessness contract); prints acceptance and both speeds.
//
//   nslm-mova-plcheck --model DIR [--res out/res] --prompts PRE [--n 256] [--only id,id]
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "engine_api.h"

static const char* opt(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char** argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);
        const char* pre = opt(argc, argv, "--prompts", NULL);
        const int n = atoi(opt(argc, argv, "--n", "256"));
        EngOpts o;
        memset(&o, 0, sizeof o);
        o.model_dir = opt(argc, argv, "--model", NULL);
        o.resource_dir = opt(argc, argv, "--res", "out/res");
        o.max_seqs = 1;
        o.kv_tokens = 20480;
        NSData* idsd = [NSData dataWithContentsOfFile:[NSString stringWithFormat:@"%s.ids", pre]];
        NSString* idx = [NSString stringWithContentsOfFile:[NSString stringWithFormat:@"%s.index", pre] encoding:NSUTF8StringEncoding error:nil];
        if (!o.model_dir || !idsd || !idx) { fprintf(stderr, "usage: nslm-mova-plcheck --model DIR --prompts PRE [--n 256]\n"); return 2; }
        NSArray* only = opt(argc, argv, "--only", NULL) ? [@(opt(argc, argv, "--only", "")) componentsSeparatedByString:@","] : nil;
        char err[512] = "";
        Eng* e = eng_open(&o, err, sizeof err);
        if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
        printf("%s\n", eng_describe(e));
        const int32_t* all = (const int32_t*) idsd.bytes;
        int32_t* a = malloc(4 * (size_t) n), *b = malloc(4 * (size_t) n);
        size_t off = 0;
        int prompts = 0, diff = 0;
        double ta = 0, tb = 0;
        long prop = 0, acc = 0, fw = 0;
        int seq = 0;
        for (NSString* line in [idx componentsSeparatedByString:@"\n"]) {
            NSArray* f = [line componentsSeparatedByString:@" "];
            if (f.count < 3) continue;
            NSString* pid = f[0];
            const int np = [f[1] intValue];
            const int32_t* ids = all + off;
            off += (size_t) np;
            if (only && ![only containsObject:pid]) continue;
            if (eng_prefill(e, 0, ids, np)) return 1;
            double t0 = now_s();
            if (eng_generate(e, &seq, 1, n, ENG_MODE_AR, a, NULL)) return 1;
            const double da = now_s() - t0;
            if (eng_prefill(e, 0, ids, np)) return 1;
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
            printf("%-8s %s  AR %5.1f tok/s  PL %5.1f tok/s (x%.2f)  accepted %4lld / %4lld  passes %4lld%s\n", pid.UTF8String,
                   k < n ? "DIFFER" : "same  ", n / da, n / db, da / db, st.accepted, st.proposals, st.forwards,
                   k < n ? [NSString stringWithFormat:@"  first difference at token %d", k].UTF8String : "");
        }
        printf("plcheck: %d prompts, %d differ; AR %.1f tok/s, PL %.1f tok/s (x%.2f); accepted %ld of %ld proposals; %.2f tokens per pass\n",
               prompts, diff, prompts * n / ta, prompts * n / tb, ta / tb, acc, prop, (double) prompts * n / fw);
        eng_close(e);
        return diff != 0;
    }
}
